# Decisions rate-limit diagnosis correction

The replacement user-provided official project key returned HTTP 200 from `https://api.openai.com/v1/decisions` at 2026-10-12T12:59:43Z. The request scored an existing unscored real candidate video and saved its official receipt; it was not a synthetic availability check.

Actual new-key headers: 5000 request limit, 2000000 token limit, 4999 remaining requests, 1999056 remaining tokens; request reset 12ms, token reset 28ms. These match the user's displayed 5000 RPM / 2M TPM limits. Request ID: `req_ad3fc7045fc34a1f819b63bf65b004b4`.

Previous requests with the earlier key returned HTTP 429, `rate_limit_exceeded`, a request limit of 50, remaining requests 0, and Retry-After 1728 seconds. Previous reports labeled this 50 RPD. That must not be generalized to the user's current organization or all Decisions access. A difference in key-associated organization/project access is possible; the exact cause is not established by the retained response headers. The new key works, so the model is not generally limited to 50 calls/day.

No API key is stored in this report, source code, or scoring receipts. Pure Decisions RL resumes the saved step-1 checkpoint with the same optimizer/RNG, reward, data and training configuration. Existing one-step results and previous stop evidence were archived before resumption. Model quality remains a separate question evaluated against the original SFT.
