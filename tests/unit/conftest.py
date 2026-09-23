import os


# Unit tests must not inherit an invalid or environment-specific development
# value from the project's local .env file. Environment variables take
# precedence over the dotenv source used by Settings.
os.environ["DEBUG"] = "false"
