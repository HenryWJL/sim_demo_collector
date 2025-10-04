# robosuite_key = key
#             # Robomimic treats 'object-state' as 'object', and so does Diffusion Policy
#             if key == 'object':
#                 robosuite_key = 'object-state'
#             assert robosuite_key in raw_obs, f"Key {key} not found in observation {raw_obs.keys()}"
#             obs[key] = np.array(raw_obs[robosuite_key])