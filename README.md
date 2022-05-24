# local-ml-environment

Template for local experimentation with machine learning.
We use hydra to manage hyperparameters and mlflow to manage experiments.
You will be notified via Slack when the experiment is completed.

# build environment

```bash
poetry install
```

and, please create .env file
```bash
touch .env
```

write SLACK_WEBHOOK_URL='YOUR SLACK WEBHOOK URL' in .env file.
