from matplotlib.pyplot import text
from slackweb import Slack


class slack_notification:
    def __init__(self, slack_webhook_url: str) -> None:

        self.slack_webhook_url = slack_webhook_url
        self.slack = Slack(url=self.slack_webhook_url)

    def send_message(self, message: str = "debug") -> None:
        self.slack.notify(text=message)
