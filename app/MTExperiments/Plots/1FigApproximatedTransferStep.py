## Imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

# Import discrete controllers
import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from controllers import *

# Import scipy signal for continuous controllers
from scipy import signal

## Set figure size as seen here
# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

## Multiplot Setup

plt.rcParams.update({'font.size': 12})


fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
fig.suptitle("Approximated Transfer Function Step Response", fontsize=16)

labels = [
    "Continuous Transfer Element Output",
    "Approximated Transfer Element Output",
    "Input Step Signal"
]

linestyles = [":", "-", "-"]

colours = ["blue", "tomato", "gray"]

alphas = [1.0, 0.6, 0.2]

# Set simulation time and discretisation
simTime = 45
ts = 1/100
samples = int(simTime*(1/ts))
# Set up discrete time and input signals
timeDisc = [ i*ts for i in range(samples) ]
inSignal = [0] + [ 1 for _ in range(samples-1) ]

#ax0.set_xlabel("Time[s]")
#ax1.set_xlabel("Time[s]")
ax2.set_xlabel("Time [sec]")

ax0.set_ylabel("T1 Open Loop")
ax1.set_ylabel("D-T1 Open Loop")
ax2.set_ylabel("PID-T1 Closed Loop")

#ax0.set_title("PT1 Open Loop")
#ax1.set_title("D-T1 Open Loop")
#ax2.set_title("PID-T1 Closed Loop")

###################################################
# PT1 Open Loop Response

## Continuous step response
def PT1Cont(kp=1, T1=1):
    return signal.lti([0, kp],[T1, 1])

timeCont, resultCont = signal.step(
    PT1Cont(kp=1, T1=5),
    T=timeDisc
)
# Plot continuous
ax0.plot(timeCont, resultCont, label=labels[0], linestyle=linestyles[0], color=colours[0], alpha=alphas[0])

## Discrete step response
# Calculate discrete output signal
block = PT1Block(kp=1, T1=5, ts=ts)
resultDisc = [ block.update(s) for s in inSignal ]
# Plot discrete
ax0.plot(timeDisc, resultDisc, label=labels[1], linestyle=linestyles[1], color=colours[1], alpha=alphas[1])
ax0.plot(timeDisc, inSignal, label=labels[2], linestyle=linestyles[2], color=colours[2], alpha=alphas[2])


###################################################
# D-T1 OpenLoop Response

## Continuous step response
def D_T1Cont(kd=1, T1=1):
    return signal.lti([kd, 0],[T1, 1])

timeCont, resultCont = signal.step(
    D_T1Cont(kd=1, T1=1),
    T=timeDisc
)
# Plot continuous
ax1.plot(timeCont, resultCont, label=labels[0], linestyle=linestyles[0], color=colours[0], alpha=alphas[0])

## Discrete step response
# Calculate discrete output signal
block = DBlock(kd=1, ts=ts)
t1 = PT1Block(kp=1, T1=1, ts=ts)
resultDisc = [ t1.update(block.update(s)) for s in inSignal ]

# Plot discrete
ax1.plot(timeDisc, resultDisc, label=labels[1], linestyle=linestyles[1], color=colours[1], alpha=alphas[1])
ax1.plot(timeDisc, inSignal, label=labels[2], linestyle=linestyles[2], color=colours[2], alpha=alphas[2])


###################################################
# PID-T1 ClosedLoop Response


## Continuous step response
def PID_T1ContClosed(kp=1, ki=1, kd=1, T1=1):
    return signal.lti([kd, kp, ki],[ T1+kd, kp+1, ki])

timeCont, resultCont = signal.step(
    PID_T1ContClosed(kp=1, ki=1, kd=1, T1=5),
    T=timeDisc
)
# Plot continuous
ax2.plot(timeCont, resultCont, label=labels[0], linestyle=linestyles[0], color=colours[0], alpha=alphas[0])

# Calculate output signal
block = PIDController(kp=1, ki=1, kd=1, ts=ts)
t1 = PT1Block(kp=1, T1=5, ts=ts)
resultDisc = []
feedback = 0
for s in inSignal:
    resultDisc.append(feedback)
    feedback = t1.update(block.update(s-feedback))

# Plot discrete
ax2.plot(timeDisc, resultDisc, label=labels[1], linestyle=linestyles[1], color=colours[1], alpha=alphas[1])
ax2.plot(timeDisc, inSignal, label=labels[2], linestyle=linestyles[2], color=colours[2], alpha=alphas[2])




ax0.legend()
ax1.legend()
ax2.legend()

set_size(7,5)

plt.tight_layout()

#plt.savefig("/app/resultsApproximatedTransferFunctionStep.pdf")
plt.show()

