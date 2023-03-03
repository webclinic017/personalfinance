import requests
import re

# Define a function to send an email with the results
def send_email(dates,ticker):
    # Set the email parameters
    sender = "your_email@example.com"
    password = "your_email_password"
    recipient = "recipient_email@example.com"
    subject = "RSI below 10"
    body = f"The RSI for {ticker} dropped below 10 on the following dates: {dates}"
    # Create the email message
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    # Send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, message.as_string())

