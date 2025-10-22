# Contributing to LLM Safety Evaluation Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- Report bugs and issues
- Suggest new features or improvements
- Improve documentation
- Submit bug fixes or feature implementations
- Add new evaluation metrics
- Implement additional defense mechanisms

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/llm-safety-evaluation.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear messages: `git commit -m "Add: brief description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Submit a pull request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/llm-safety-evaluation.git
cd llm-safety-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy

# Run tests
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Formatting

Use `black` for code formatting:

```bash
black src/ tests/
```

### Linting

Use `flake8` for linting:

```bash
flake8 src/ tests/
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high code coverage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Generate coverage report
pytest --cov=src --cov-report=html tests/
```

## Documentation

- Update README.md for major changes
- Add docstrings to new functions/classes
- Update relevant documentation in docs/
- Include usage examples

## Commit Messages

Use clear, descriptive commit messages:

- `Add: new feature or file`
- `Update: changes to existing feature`
- `Fix: bug fixes`
- `Docs: documentation changes`
- `Test: test additions or changes`
- `Refactor: code refactoring`

## Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG if applicable
5. Request review from maintainers
6. Address review feedback promptly

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect differing opinions and experiences

## Questions?

Feel free to open an issue for questions or clarifications.

Thank you for contributing!
