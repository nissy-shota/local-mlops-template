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
SLACK_WEBHOOK_URL='YOUR SLACK WEBHOOK URL'
HOME_PC_NAME="YOUR HOME PC NODE NAME"
UNIV_PC_NAME="YOUR UNIV PC NODE NAME"
```
