def test_allocator_exposes_one_to_one_strike_tasks(
    sample_context,
    milp_ok,
):
    for target in sample_context.world_state["targets"]:
        target["confirmed"] = True

    result = milp_ok.allocate_action(sample_context)

    assert result.success, result.reason
    platforms = [task.platform for task in result.data]
    targets = [task.target for task in result.data]
    assert len(platforms) == len(set(platforms))
    assert len(targets) == len(set(targets))
    assert len(result.data) == 2
