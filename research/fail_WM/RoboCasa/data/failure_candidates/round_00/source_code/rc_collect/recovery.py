def retry_after_measured_failure(controller):
    e=controller.env
    grasp=bool(e._check_grasp(e.robots[0].gripper['right'],e.objects['cube']))
    lifted=bool(controller.cube()[2] > 1.0)
    if not (grasp and lifted):
        controller.hold(controller.eef(),-1,25,'recovery_pause')
        # Uses the actual displaced object position; no reset or attachment.
        controller.attempt()
