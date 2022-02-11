import matplotlib.pyplot as plt
import numpy as np

def clampValue(val, valMax):
    """ Makes sure that a value is within [-valMax, valMax] """
    if valMax == None: return val
    if val > valMax: return valMax
    elif val < -1*valMax: return -1*valMax
    else: return val

class PIDController:
    """ Discrete PID controller approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=0, ki=0, kd=0, ts=0, feedforward=0, bufferLength=3) -> None:
        #self.bufferLength = bufferLength
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        self.feedforward = feedforward
        if ts != 0:  self.setConstants(kp, ki, kd, ts)
    #
    def setConstants(self, kp, ki, kd, ts):
        self.k1 = kp+((ki*ts)/2)+((2*kd)/ts)
        self.k2 = ki*ts-((4*kd)/ts)
        self.k3 = (-1*kp)+((ki*ts)/2)+((2*kd)/ts)   
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k1 + e[1]*self.k2 + e[2]*self.k3 + y[2]
        return y[0] + e[0]*self.feedforward

class PT1Block:
    """ Discrete PT1 block approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=1, T1=0, ts=0, bufferLength=2) -> None:
        self.k1 = 0
        self.k2 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kp, T1, ts)
    #
    def setConstants(self, kp, T1, ts):
        t = 2*(T1/ts)
        self.k1 = kp/(1+t)
        self.k2 = (1-t)/(1+t)
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = (e[0] + e[1])*self.k1 - y[1]*self.k2
        return y[0]

class DBlock:
    def __init__(self, kd=0, ts=0, bufferLength=2) -> None:
        self.k = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kd, ts)
    #
    def setConstants(self, kd, ts):
        self.k = (2*kd)/ts
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k - e[1]*self.k - y[1]
        return y[0]

class strategy4Controller:
    """ Models HEBI control strategy 4 using 3 PID controllers discretised using the tustin approximation """
    def __init__(self, ts=0) -> None:
        """ Class Constructor """
        self.ts = ts
        # Instantiate PIDControllers: Param order is [PositionPID, VelocityPID, EffortPID]
        # https://docs.hebi.us/core_concepts.html#control-strategies
        # Default controller params:
        # https://docs.hebi.us/resources/gains/X5-4_STRATEGY4.xml
        self.PositionPID = PIDController()
        self.VelocityPID = PIDController(feedforward=1)
        self.EffortPID   = PIDController(feedforward=1)
        self.EffortD     = DBlock()
        # Use in- and output filters to mimick motor constraints
        # Order is [pos, vel, effort] and params are for the X4-5 model
        self.inputFilters = [
            PT1Block(kp=1, T1=0, ts=ts),
            PT1Block(kp=1, T1=0, ts=ts),
            PT1Block(kp=1, T1=0, ts=ts)
        ]
        self.outputFilters = [
            PT1Block(kp=1, T1=0,    ts=ts),
            PT1Block(kp=1, T1=0.75, ts=ts),
            PT1Block(kp=1, T1=0.9,  ts=ts)
        ]
    #
    def updateConstants(self, constants):
        """ Calculates constants in each PID controller """
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(  constants[6], constants[7], 0,            self.ts)
        self.EffortD.setConstants(constants[8],                                 self.ts)
    #
    def update(self, vecIn, feedback, targetConstraints=[None, 3.43, 20], outputConstraints=[10, 1, 1]):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        # All signal vectors are of form [Position, Velocity, Effort]
        # The values are clamped at the input and output of each PID controller, like in the Hebi implementation
        # Clamp input values
        vecIn = [ clampValue(entry, valMax) for entry, valMax in zip(vecIn, targetConstraints) ]
        # Apply input T1 filters to input values 
        vecIn = [ block.update(entry) for entry, block in zip(vecIn, self.inputFilters) ]
        #
        # Update Position PID (input = positionIn - positionFeedback), apply output filtering and clamp output value
        effort = self.PositionPID.update(vecIn[0] - feedback[0])
        effort = self.outputFilters[0].update(effort)
        effort = clampValue(effort, outputConstraints[0])
        # Update Effort PID (input = yPositionPID + effortIn - effortFeedback), apply output filtering and clamp output value
        # The parallel effort D controller is added here as well without subtracting the feedback
        PWM1 = self.EffortPID.update(effort + vecIn[2] - feedback[2]) + self.EffortD.update(effort + vecIn[2])
        PWM1 = self.outputFilters[2].update(PWM1)
        PWM1 = clampValue(PWM1, outputConstraints[2])
        # Update Velocity PID (input = velocityIn - velocityFeedback) , apply output filtering and clamp output value
        PWM2 = self.VelocityPID.update(vecIn[1] - feedback[1])
        PWM2 = self.outputFilters[1].update(PWM2)
        PWM2 = clampValue(PWM2, outputConstraints[1])
        # Return sum of PWM signals
        return PWM1 + PWM2

class strategy4Controller:
    """ Models HEBI control strategy 4 using 3 PID controllers discretised using the tustin approximation """
    def __init__(self, ts=0) -> None:
        """ Class Constructor """
        self.ts = ts
        # Instantiate PIDControllers: Param order is [PositionPID, VelocityPID, EffortPID]
        # https://docs.hebi.us/core_concepts.html#control-strategies
        # Default controller params:
        # https://docs.hebi.us/resources/gains/X5-4_STRATEGY4.xml
        self.PositionPID = PIDController()
        self.VelocityPID = PIDController(feedforward=1)
        self.EffortPID = PIDController(feedforward=1)
        # Use in- and output filters to mimick motor constraints
        # Order is [pos, vel, effort] and params are for the X4-5 model
        self.inputFilters = [
            PT1Block(kp=1, T1=0, ts=ts),
            PT1Block(kp=1, T1=0, ts=ts),
            PT1Block(kp=1, T1=0, ts=ts)
        ]
        self.outputFilters = [
            PT1Block(kp=1, T1=0,    ts=ts),
            PT1Block(kp=1, T1=0.75, ts=ts),
            PT1Block(kp=1, T1=0.9,  ts=ts)
        ]
    #
    def updateConstants(self, constants):
        """ Calculates constants in each PID controller """
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(constants[6], constants[7], constants[8], self.ts)
    #
    def update(self, vecIn, feedback, targetConstraints=[None, 3.43, 20], outputConstraints=[10, 1, 1]):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        # All signal vectors are of form [Position, Velocity, Effort]
        # The values are clamped at the input and output of each PID controller, like in the Hebi implementation
        # Clamp input values
        vecIn = [ clampValue(entry, valMax) for entry, valMax in zip(vecIn, targetConstraints) ]
        # Apply input T1 filters to input values 
        vecIn = [ block.update(entry) for entry, block in zip(vecIn, self.inputFilters) ]
        #
        # Update Position PID (input = positionIn - positionFeedback), apply output filtering and clamp output value
        effort = self.outputFilters[0].update( self.PositionPID.update(vecIn[0] - feedback[0]) )
        effort = clampValue(effort, outputConstraints[0])
        # Update Effort PID (input = yPositionPID + effortIn - effortFeedback), apply output filtering and clamp output value
        PWM1 = self.outputFilters[2].update( self.EffortPID.update(effort + vecIn[2] - feedback[2]) )
        PWM1 = clampValue(PWM1, outputConstraints[2])
        # Update Velocity PID (input = velocityIn - velocityFeedback) , apply output filtering and clamp output value
        PWM2 = self.outputFilters[1].update( self.VelocityPID.update(vecIn[1] - feedback[1]) )
        PWM2 = clampValue(PWM2, outputConstraints[1])
        # Return sum of PWM signals
        return PWM1 + PWM2




### TEST D ###
block = DBlock(kd=1, ts=1/60)
t1 = PT1Block(kp=1, T1=5, ts=1/60)
simTime = 60
inSignal = 1
res = [ t1.update(block.update(inSignal)) for i in range(60*simTime) ]
time = [ i/60 for i in range(0, 60*simTime) ]
plt.plot(time, res)
plt.show()


### TEST PIDT1  ###
block = PIDController(kp=1, ki=1, kd=1, ts=1/60)
t1 = PT1Block(kp=1, T1=5, ts=1/60)
res = []
inSignal = 1
feedback = 0
simTime = 60
for i in range(60*simTime):
    res.append(feedback)
    feedback = t1.update(block.update( inSignal - feedback ))

time = [ i/60 for i in range(0, 60*simTime) ]
plt.plot(time, res)
plt.show()


### TEST T1 ###
t1 = PT1Block(kp=1, T1=5, ts=1/60)
simTime = 60
inSignal = 1
res = [ t1.update(inSignal) for i in range(60*simTime) ]
time = [ i/60 for i in range(0, 60*simTime) ]
plt.plot(time, res)
plt.show()


###  TEST STRATEGY4 CONTROLLER  ###
block = strategy4Controller(ts=1/60)
block.updateConstants(
    [30, 0, 0, 0.05, 0, 0, 0.25, 0, 0.001]
)
simTime = 30
inSignal = [1, 1, 1]
feedback = [0, 0, 0]
res = [ block.update(inSignal, feedback) for i in range(60*simTime) ]
time = [ i/60 for i in range(0, 60*simTime) ]
plt.plot(time, res)
plt.show()