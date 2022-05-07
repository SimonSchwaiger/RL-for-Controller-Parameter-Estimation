
from multiprocessing import shared_memory
import re

class sharedMemJointMetric:
    """ 
    Shared memory communication between physics server and gym environment

    Server initialises shared memory and waits for "ready" to be true.
    The environment is registered and a physics update is performed.
    The server only processes data when ready is set to True, while the env only processes data when ready is set to False
    """
    def __init__(self, jointID, server=False, sharedMemSize=250) -> None:
        # Define jointmetric attributes
        self.ready: bool
        self.registered: bool
        self.jointID: int
        self.updateCmd: str
        self.feedbackCmd: str
        self.jointFeedback: str
        #
        # Set up shared mem, based on being server
        self.sharedMemSize = sharedMemSize
        #
        shmName = "Joint{}PhysicsSimSharedMem".format(jointID)
        if server:
            self.shm = shared_memory.SharedMemory(create=True, size=self.sharedMemSize, name=shmName)
        else:
            self.shm = shared_memory.SharedMemory(shmName)
        #
        # Compiled regex for cleaning up received messages
        self.regex = re.compile('\x00')
        #
        # Track if we are the server
        self.server = server
        # If we are the server, format empty state, to track active envs
        if server == True: 
            self.setState(
                False,
                False,
                jointID,
                " ",
                " ",
                " "
            )
            self.flushState()
    #   
    def __del__(self):
        self.shm.close()
        if self.sercer: self.shm.unlink()
    #
    def unregister(self):
        self.shm.close()
        if self.server: self.shm.unlink()
    #
    def setState(self, ready, registered, jointID, updateCmd, feedbackCmd, jointFeedback):
        self.ready = ready
        self.registered = registered
        self.jointID = jointID
        self.updateCmd = updateCmd
        self.feedbackCmd = feedbackCmd
        self.jointFeedback = jointFeedback
    #
    def flushState(self):
        # Load current message to determine its length
        msgOld = bytes(self.shm.buf[:self.sharedMemSize]).decode("utf-8")
        msgOld = re.sub(self.regex, '', msgOld)
        #
        # Format new message
        if self.ready == True: ready = 1
        else: ready = 0
        if self.registered == True: registered = 1
        else: registered = 0
        msg = "{}\n{}\n{}\n{}\n{}\n{}".format(
            ready,
            registered,
            self.jointID,
            self.updateCmd,
            self.feedbackCmd,
            self.jointFeedback
        ).encode()
        #
        # Reset bytes that are not needed anymore
        if len(msg) < len(msgOld):
            self.shm.buf[len(msg):len(msgOld)] = '\x00'
        #
        # Write message to buffer
        self.shm.buf[1:len(msg)] = msg[1:]
        # Write ready last for synchronisation
        self.shm.buf[:1] = msg[:1]
    #
    def loadState(self):
        msg = bytes(self.shm.buf[:self.sharedMemSize]).decode("utf-8")
        msg = re.sub(self.regex, '', msg)
        msg = msg.split("\n")
        self.ready = int(msg[0]) == 1
        self.registered = int(msg[1]) == 1
        self.jointID = int(msg[2])
        self.updateCmd = msg[3]
        self.feedbackCmd = msg[4]
        self.jointFeedback = msg[5]
    #
    def getState(self):
        self.loadState()
        return {
            "ready": self.ready,
            "registered": self.registered,
            "jointID": self.jointID,
            "updateCmd": self.updateCmd,
            "feedbackCmd": self.feedbackCmd,
            "jointFeedback": self.jointFeedback
        }
    #
    def checkRegistered(self):
        return int(bytes(self.shm.buf[2:3]).decode("utf-8")) == 1
    #
    def checkReady(self):
        return int(bytes(self.shm.buf[:1]).decode("utf-8")) == 1


