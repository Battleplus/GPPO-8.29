#pragma once

#include "brain_cpp/algorithm_interfaces.hpp"
#include "brain_cpp/mission_context.hpp"
#include "brain_cpp/mission_event.hpp"

namespace brain_cpp {

class MissionFSM {
public:
    MissionFSM(
        MissionContext& context,
        ITaskAllocator& taskAllocator,
        IRoutePlanner& routePlanner,
        IPositionSelector& positionSelector);

    MissionState dispatch(const MissionEvent& event);
    MissionState currentState() const;

private:
    MissionContext& ctx_;
    ITaskAllocator& taskAllocator_;
    IRoutePlanner& routePlanner_;
    IPositionSelector& positionSelector_;

    MissionState runAutoChain();
    void transitionTo(MissionState nextState, const MissionEvent* event = nullptr);

    MissionState doReconAllocate();
    MissionState doReconPlan();
    MissionState doUpdateWorld();
    MissionState doActionAllocate();
    MissionState doPositionSelect();
    MissionState doActionPlan();
    MissionState doReplan();
};

}  // namespace brain_cpp
