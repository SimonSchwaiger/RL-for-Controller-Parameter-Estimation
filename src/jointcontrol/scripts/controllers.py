#!/usr/bin/env python3

""" These classes implement discrete approximations of the tested controllers. """

def clampValue(val, valMax):
    """ Makes sure that a value is within [-valMax, valMax] """
    if valMax == None: return val
    if val > valMax: return valMax
    elif val < -1*valMax: return -1*valMax
    else: return val

class PIDController:
    """!@brief Discrete PID controller approximated using the Tustin (trapezoid) approximation """
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
        """ Updates controller constants """
        self.k1 = kp+((ki*ts)/2)+((2*kd)/ts)
        self.k2 = ki*ts-((4*kd)/ts)
        self.k3 = (-1*kp)+((ki*ts)/2)+((2*kd)/ts)   
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """
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
    """!@brief Discrete PT1 block approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=1, T1=0, ts=0, bufferLength=2) -> None:
        self.k1 = 0
        self.k2 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kp, T1, ts)
    #
    def setConstants(self, kp, T1, ts):
        """ Updates controller constants """
        t = 2*(T1/ts)
        self.k1 = kp/(1+t)
        self.k2 = (1-t)/(1+t)
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
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
    """!@brief Discrete D Block approximated using the Tustin approximation """
    def __init__(self, kd=0, ts=0, bufferLength=2) -> None:
        self.k = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kd, ts)
    #
    def setConstants(self, kd, ts):
        """ Updates controller constants """
        self.k = (2*kd)/ts
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        if self.k == 0: return 0
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k - e[1]*self.k - y[1]
        return y[0]

class PT2Block:
    """!@brief Discrete PT2 Block approximated using the Tustin approximation """
    def __init__(self, T=0, D=0, kp=1, ts=0, bufferLength=3) -> None:
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0:  self.setConstants(T, D, kp, ts)
    #
    def setConstants(self, T, D, kp, ts):
        """ Updates controller constants """
        self.k1 = 4*T**2 + 4*D*T*ts + ts**2
        self.k2 = 2*ts**2 - 8*T**2
        self.k3 = 4*T**2 - 4*D*T*ts + ts**2
        self.k4 = kp*ts**2
        self.k5 = 2*kp*ts**2
        self.k6 = kp*ts**2
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = ( e[0]*self.k4 + e[1]*self.k5 + e[2]*self.k6 - y[1]*self.k2 - y[2]*self.k3 )/self.k1
        return y[0]

class smartPID:
    """!@brief Implementation to approximate HEBI PID controller behaviour. Variable names are consistent with HEBI Gains format 
    
    This approximation of HEBI controller and gains is based on their documentation of control strategy 4 (1) and controller gains documentation (2)
    1) https://docs.hebi.us/core_concepts.html#control-strategies
    2) https://docs.hebi.us/core_concepts.html#controller_gains 
    """
    def __init__(self, kp=0, ki=0, kd=0, targetLP=0, outputLP=0, ts=0, feedforward=0, d_on_error=True, targetMax=None, outputMax=None) -> None:
        self.d_on_error = d_on_error
        if d_on_error:
            self.PID = PIDController(kp=kp, ki=ki, kd=kd, ts=ts)
            self.D = DBlock(kd=0, ts=ts)
        else:
            self.PID = PIDController(kp=kp, ki=ki, kd=0, ts=ts)
            self.D = DBlock(kd=kd, ts=ts)
        #
        self.targetLP = targetLP
        self.outputLP = outputLP
        self.inputFilter  = PT1Block(kp=1, T1=targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=outputLP, ts=ts)
        #
        self.targetMax = targetMax
        self.outputMax = outputMax
        self.feedforward = feedforward
    #
    def setConstants(self, kp, ki, kd, ts):
        """ Updates controller constants """
        if self.d_on_error:
            self.PID.setConstants(kp, ki, kd, ts)
        else:
            self.PID.setConstants(kp, ki, 0, ts)
            self.D.setConstants(kd, ts)
        #
        self.inputFilter  = PT1Block(kp=1, T1=self.targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=self.outputLP, ts=ts)
    #
    def update(self, target, feedback):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        # Clamp and low pass input
        filteredInput = self.inputFilter.update(
            clampValue(target, self.targetMax)
        )
        # Update internal control blocks
        output = self.D.update(filteredInput) + self.PID.update(filteredInput - feedback)
        # Clamp and low pass output
        output = self.outputFilter.update(
            clampValue(output + self.feedforward*filteredInput, self.outputMax)
        )
        return output
        
class strategy4Controller:
    """!@brief Models HEBI control strategy 4 using 3 PID controllers discretised using the Tustin approximation 
    
    This approximation of the HEBI controller is based on HEBI's documentation: https://docs.hebi.us/core_concepts.html#control-strategies
    """
    def __init__(self, ts=0, targetConstraints=[None, 3.43, 20], outputConstraints=[10, 1, 1], feedfowards=[0, 1, 1], d_on_errors=[True, True, False], constants=None) -> None:
        """ Class constructor """
        self.ts = ts
        #
        self.PositionPID = smartPID(
            targetMax=targetConstraints[0],
            outputMax=outputConstraints[0],
            feedforward=feedfowards[0],
            d_on_error=d_on_errors[0]
        )
        #
        self.VelocityPID = smartPID(
            targetMax=targetConstraints[1],
            outputMax=outputConstraints[1],
            feedforward=feedfowards[1],
            d_on_error=d_on_errors[1],
            outputLP=0.01
        )
        #
        self.EffortPID = smartPID(
            targetMax=targetConstraints[2],
            outputMax=outputConstraints[2],
            feedforward=feedfowards[2],
            d_on_error=d_on_errors[2],
            outputLP=0.001
        )
        #
        self.PWMFilter = PT2Block(kp=1, T=0.0, D=10, ts=self.ts)
        if constants != None: self.updateConstants(constants)
    #
    def updateConstants(self, constants):
        """ Updates controller constants """
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(  constants[6], constants[7], constants[8], self.ts)
    #
    def update(self, vecIn, feedback):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        effort = self.PositionPID.update(vecIn[0], feedback[0])
        PWM1 = self.EffortPID.update(vecIn[2] + effort, feedback[2])
        PWM2 = self.VelocityPID.update(vecIn[2], feedback[2])
        return self.PWMFilter.update(PWM1 + PWM2)

def PWM2Torque(PWM, maxMotorTorque=7.5):
    """ Converts PWM output signals of the strategy 4 controller to direct torque output (Nm) """
    # PWM range -> [-1, 1], Since we have X5-4 motors, the max torque is 7.5 Nm
    # The conversion is assumed to be linear for this implementation, this should yield enough accuracy for the tested scenarios
    return PWM*maxMotorTorque

def deserialiseJointstate(js):
    """ Unpacks JointState type to list of [position, velocity, effffort] for each joint """
    return [ [pos, vel, eff] for pos, vel, eff in zip(js.position, js.velocity, js.effort) ]