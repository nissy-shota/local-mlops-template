import os
from platform import node

from matplotlib.pyplot import text
from slackweb import Slack


class slack_notification:
    def __init__(self, slack_webhook_url: str, home: str, univ: str) -> None:

        self.slack_webhook_url = slack_webhook_url
        self.slack = Slack(url=self.slack_webhook_url)

        node_name = os.uname().nodename

        if node_name == home:
            self.place = "home"
        elif node_name == univ:
            self.place = "univ"
        else:
            self.place = "where?"

    def send_message(self, message: str = "debug") -> None:

        message += f" running in {self.place}"
        self.slack.notify(text=message)
