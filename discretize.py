class FineTuning:
  """Fine tuning options, aka policy post-processing.
  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.
  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.
  The post-processing is used on the learner, and thus must be jit-friendly.
  """
  # The learner step after which the policy post processing (aka finetuning)
  # will be enabled when learning. A strictly negative value is equivalent
  # to infinity, ie disables finetuning completely.
  from_learner_steps: int = -1
  # All policy probabilities below `threshold` are zeroed out. Thresholding
  # is disabled if this value is non-positive.
  policy_threshold: float = 0.03
  # Rounds the policy probabilities to the "closest"
  # multiple of 1/`self.discretization`.
  # Discretization is disabled for non-positive values.
  policy_discretization: int = 32

  def __call__(self, policy: chex.Array, mask: chex.Array,
               learner_steps: int) -> chex.Array:
    """A configurable fine tuning of a policy."""
    chex.assert_equal_shape((policy, mask))
    do_finetune = jnp.logical_and(self.from_learner_steps >= 0,
                                  learner_steps > self.from_learner_steps)

    return jnp.where(do_finetune, self.post_process_policy(policy, mask),
                     policy)

  def post_process_policy(
      self,
      policy: chex.Array,
      mask: chex.Array,
  ) -> chex.Array:
    """Unconditionally post process a given masked policy."""
    chex.assert_equal_shape((policy, mask))
    policy = self._threshold(policy, mask)
    policy = self._discretize(policy)
    return policy

  def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    chex.assert_equal_shape((policy, mask))
    if self.policy_threshold <= 0:
      return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= self.policy_threshold) +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold))
    return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, policy: chex.Array) -> chex.Array:
    """Round all action probabilities to a multiple of 1/self.discretize."""
    if self.policy_discretization <= 0:
      return policy

    # The unbatched/single policy case:
    if len(policy.shape) == 1:
      return self._discretize_single(policy)

    # policy may be [B, A] or [T, B, A], etc. Thus add hk.BatchApply.
    dims = len(policy.shape) - 1

    # TODO(author18): avoid mixing vmap and BatchApply since the two could
    # be folded into either a single BatchApply or a sequence of vmaps, but
    # not the mix.
    vmapped = jax.vmap(self._discretize_single)
    policy = hk.BatchApply(vmapped, num_dims=dims)(policy)

    return policy

  def _discretize_single(self, mu: chex.Array) -> chex.Array:
    """A version of self._discretize but for the unbatched data."""
    # TODO(author18): try to merge _discretize and _discretize_single
    # into one function that handles both batched and unbatched cases.
    if len(mu.shape) == 2:
      mu_ = jnp.squeeze(mu, axis=0)
    else:
      mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32)
    result = jnp.zeros_like(mu_)
    order = jnp.argsort(-mu_)  # Indices of descending order.
    weight_left = self.policy_discretization

    def f_disc(i, order, roundup, weight_left, result):
      x = jnp.minimum(roundup[order[i]], weight_left)
      result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x),
                               result)
      weight_left -= x
      return i + 1, order, roundup, weight_left, result

    def f_scan_scan(carry, x):
      i, order, roundup, weight_left, result = carry
      i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
          i, order, roundup, weight_left, result)
      carry_next = (i_next, order_next, roundup_next, weight_left_next,
                    result_next)
      return carry_next, x

    (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
        f_scan_scan,
        init=(jnp.asarray(0), order, roundup, weight_left, result),
        xs=None,
        length=n_actions)

    result_next = jnp.where(weight_left_next > 0,
                            result_next.at[order[0]].add(weight_left_next),
                            result_next)
    if len(mu.shape) == 2:
      result_next = jnp.expand_dims(result_next, axis=0)
    return result_next / self.policy_discretization