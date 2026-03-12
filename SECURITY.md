# Security Policy

## Supported Versions

This is an academic research project. Only the latest version on the `main`
branch is actively maintained.

| Version | Supported |
|---------|-----------|
| main    | ✅        |

## Reporting a Vulnerability

This project does not handle sensitive data, user credentials, or network
services. However, if you discover a security issue (e.g., a dependency with
a known CVE), please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly at **n.allerponte@udc.es**
3. Include a description of the vulnerability and steps to reproduce

You can expect a response within 7 days. If the issue is confirmed, a fix
will be released as soon as possible.

## Dependencies

This project uses standard scientific Python packages (PyTorch, scikit-learn,
matplotlib). Keep dependencies up to date with:

```bash
uv sync --upgrade
```
