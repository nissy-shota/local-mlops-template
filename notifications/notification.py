import os
from platform import node

from matplotlib.pyplot import text
from slackweb import Slack


class slack_notification:
    def __init__(self, slack_webhook_url: str) -> None:

        self.slack_webhook_url = slack_webhook_url
        self.slack = Slack(url=self.slack_webhook_url)

        self.node_name = os.uname().nodename

    def send_message(self, message: str = "debug") -> None:

        message += f" running in {self.node_name}"
        self.slack.notify(text=message)
