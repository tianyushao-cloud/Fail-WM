# World-model diagnostics

Validation-only diagnostics; no training performed.

## success
- success=True, event=True, episodes=53, targets=1180, episode-mean visual NLL=1.073892611349386
- success=True, event=False, episodes=53, targets=212, episode-mean visual NLL=1.0945874236102373
- success=False, event=True, episodes=47, targets=689, episode-mean visual NLL=1.7447141461479767
- success=False, event=False, episodes=47, targets=188, episode-mean visual NLL=1.6552725413695295
- Action-swap informative targets=79, pairs=44, mean paired failure preference=0.0798274120146578, positive pairs=26
## natural
- success=True, event=True, episodes=53, targets=1180, episode-mean visual NLL=0.7499728526139025
- success=True, event=False, episodes=53, targets=212, episode-mean visual NLL=0.7173635591313524
- success=False, event=True, episodes=47, targets=689, episode-mean visual NLL=1.0484486269774254
- success=False, event=False, episodes=47, targets=188, episode-mean visual NLL=0.842820368865703
- Action-swap informative targets=79, pairs=44, mean paired failure preference=0.13258186108009382, positive pairs=37

Codec summaries:
{
  "success": {
    "episodes": 53,
    "frames": 1233,
    "episode_mean_roi_mae_255": 8.990320207854273,
    "episode_mean_center_error_px": 0.6420840658427669,
    "missed_visible_cube_frames": 12
  },
  "failure": {
    "episodes": 47,
    "frames": 736,
    "episode_mean_roi_mae_255": 11.842766994006764,
    "episode_mean_center_error_px": 3.4203798488149326,
    "missed_visible_cube_frames": 30
  },
  "contact_without_lift": {
    "episodes": 9,
    "frames": 158,
    "episode_mean_roi_mae_255": 15.486261377375747,
    "episode_mean_center_error_px": 6.670092511564173,
    "missed_visible_cube_frames": 15
  },
  "dropped_before_goal": {
    "episodes": 7,
    "frames": 144,
    "episode_mean_roi_mae_255": 14.30294714306495,
    "episode_mean_center_error_px": 2.0775003232547626,
    "missed_visible_cube_frames": 6
  },
  "grasp_miss": {
    "episodes": 15,
    "frames": 120,
    "episode_mean_roi_mae_255": 8.630049462651217,
    "episode_mean_center_error_px": 3.991600673884726,
    "missed_visible_cube_frames": 2
  },
  "lifted_but_not_placed": {
    "episodes": 15,
    "frames": 299,
    "episode_mean_roi_mae_255": 11.637186972591541,
    "episode_mean_center_error_px": 1.4853145486977337,
    "missed_visible_cube_frames": 7
  },
  "object_off_counter": {
    "episodes": 1,
    "frames": 15,
    "episode_mean_roi_mae_255": 13.104519791840122,
    "episode_mean_center_error_px": 4.030789688704059,
    "missed_visible_cube_frames": 0
  }
}

NLL does not measure optimism. RGB sampling outcomes are provisional; absent/low-area cube is indeterminate.
A lower full-image error or an occasional failed sample does not prove calibrated outcome prediction.
