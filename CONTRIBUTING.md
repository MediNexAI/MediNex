# Contributing to MediNex AI

We love your input! We want to make contributing to MediNex AI as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repository to your own GitHub account
2. Clone the project to your machine
3. Create a branch locally with a succinct but descriptive name
4. Commit changes to the branch
5. Push changes to your fork
6. Open a PR in our repository and follow the PR template

### Pull Request Guidelines

- Update the README.md with details of changes to the interface, if applicable
- Update the documentation with details of any new functionality
- The PR should work for Python 3.9 and above
- Make sure your code lints (we use flake8)
- Include appropriate tests

## Development Setup

To set up your development environment:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r ai/requirements.txt

# Install development dependencies
pip install flake8 pytest pytest-cov black
```

## Code Style

We use [black](https://github.com/psf/black) for code formatting and [flake8](https://flake8.pycqa.org/en/latest/) for linting. Please ensure your code adheres to these standards.

```bash
# Format code
black .

# Lint code
flake8 .
```

## Testing

We use pytest for testing. Make sure to add tests for any new functionality.

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=ai
```

## License

By contributing to MediNex AI, you agree that your contributions will be licensed under the project's MIT License.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## Questions?

Feel free to contact the project maintainers if you have any questions.

Thank you for your contribution! 