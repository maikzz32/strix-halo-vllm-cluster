# Publication privacy rules

Never commit or publish real deployment usernames, passwords, API keys, access
tokens, private keys, or credential-bearing URLs. This applies to documentation,
source defaults, configuration, logs, benchmark exports, and PR/issue text.

Use neutral deployment placeholders and locally supplied configuration.
`cluster-user` and `/home/cluster-user` are placeholders, not working login
settings. Do not copy sanitized defaults into an existing deployment unchanged.
Check tracked diffs and exported artifacts before every publication. Keep
credentials in environment variables or the platform secret store; references
such as `secrets.GITHUB_TOKEN` may be committed, their values may not.

Published historical measurements retain their numerical values. Deployment
usernames and paths are redacted in exported records; source/result hashes refer
to the original locally retained artifacts, not the redacted export bytes.
Do not claim that a cleanup commit removes sensitive content from git history.
