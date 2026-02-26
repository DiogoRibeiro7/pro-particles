# Contributing to pro-particles

Thank you for your interest in contributing to pro-particles!

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install Poetry if you haven't already: https://python-poetry.org/docs/#installation
4. Install dependencies:
   ```bash
   poetry install
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   poetry run ruff check .
   poetry run mypy src
   poetry run pytest -q
   ```

3. Commit your changes with a descriptive commit message

4. Push to your fork and create a pull request

## Code Standards

- Code is formatted and checked using `ruff`
- Type hints are checked using `mypy` with strict settings
- All tests must pass before merging
- New features should include appropriate tests

## Testing

Run the test suite:
```bash
poetry run pytest
```

Run with coverage:
```bash
poetry run pytest --cov=pro_particles
```

## Pull Request Process

1. Ensure all CI checks pass
2. Update the README.md if needed
3. The PR will be reviewed and merged once approved

## Questions?

Feel free to open an issue for any questions or concerns.