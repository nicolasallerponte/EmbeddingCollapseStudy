# Contributing

Thank you for your interest in contributing to EmbeddingCollapseStudy!

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or suggest improvements
- Include a clear description, steps to reproduce, and relevant logs or outputs

### Submitting Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes and commit: `git commit -m "feat: description"`
4. Push to your fork: `git push origin feat/your-feature`
5. Open a Pull Request with a clear description of your changes

### Commit Convention
We follow conventional commits:
- `feat:` new feature or experiment
- `fix:` bug fix
- `docs:` documentation changes
- `refactor:` code restructuring without behavior change
- `analysis:` new analysis or figures

### Code Style
- Python code is formatted with `ruff` (`uv run ruff format .`)
- Keep functions documented with docstrings
- New experiments should include a config in `configs/`

## Areas Where Help is Welcome
- Extending sweeps to STL-10 or ImageNet-100
- Adding new geometry metrics (e.g., intrinsic dimensionality)
- Reproducing results with other SSL methods (MoCo, BYOL, Barlow Twins)
- Improving visualizations in the analysis notebook
