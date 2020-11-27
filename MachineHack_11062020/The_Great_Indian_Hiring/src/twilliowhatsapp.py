import os
from twilio.rest import Client
from The_Great_Indian_Hiring.src.constants import constTWILIOWHATSAPPFROM, constTWILIOWHATSAPPTO, constTWILIOACCOUNTAUTH, constTWILIOACCOUNTSID


class clsTwilioWhatsapp:

    def __init__(self):
        self.whatsAPPFrom = constTWILIOWHATSAPPFROM
        self.whatsAPPTo = constTWILIOWHATSAPPTO
        self.accountSID = constTWILIOACCOUNTSID
        self.authToken = constTWILIOACCOUNTAUTH

    def sendingWhatsAppMessage(self, fncpMessage):
        """
        This function sends an message to the number saved in constTWILIOWHATSAPPTO.

            Parameters:
                fncpMessage (str) : Message to be sent

            Returns:
                success (bool): whether success or not.
        """
        client = Client(self.accountSID, self.authToken)

        client.messages \
            .create(
                 body=fncpMessage,
                 from_=self.whatsAPPFrom,
                 to=self.whatsAPPTo
             )
        return True


if __name__ == '__main__':
    whatsApp = clsTwilioWhatsapp()
    whatsApp.sendingWhatsAppMessage('Testing Message')