# Security Policy

## Public boundary

The repository must not contain:

- API keys, passwords, access keys, SSH keys, tokens, or `.env` files;
- private-network endpoints, host aliases, deployment paths, or account names;
- model weights or internal offline dependency archives;
- data that the publisher is not authorized to redistribute.

Runtime inputs and outputs belong under ignored `runtime/`, `data/`, or
`artifacts/` directories.

## Public demo

Use `scripts/build_public_demo.py` to generate a public static demo from an
accepted release. The builder removes private-network asset URLs, hides the
runtime model identifier, and fails when private IPs or credential-like values
remain.

The demo builder does not determine whether document content is legally or
organizationally publishable. The publisher remains responsible for data
authorization.

## Reporting

Do not open a public issue containing a credential or private endpoint. Report
security problems privately to the repository maintainers and include only the
minimum information required to reproduce the issue.
