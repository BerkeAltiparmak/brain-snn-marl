import numpy as np
import control

def FE_init(time,dt):
    """Initialization of Forward Euler.
    Given simulation time (seconds) and timestep size (dt), returns list of times (for looping purposes) and the number of total timesteps (Nt).

    Args:
        time (integer): the number of seconds to run the FE simulation for, in seconds.
        dt (float): the length of a single timestep, in seconds.

    Returns:
        ndarray: array of times to loop over, or for plotting purposes.
        int: the total number of timesteps in the 
    """    
    #Forward Euler parameters
    times = np.arange(0, time, dt)
    Nt=len(times)

    return times,Nt

"""
Initialization of SMD system.
Given m, k and c (SMD parameters), returns A and B matrices for SMD system in state-space form.
"""
def SMD_init(m=3,k=5,c=0.5):

    #A-matrix, defines dynamics of the DS
    A = np.array([[0,1],
            [(-k/m),(-c/m)]]) 

    #B-matrix, defines the influence of the force on the system
    B = np.array([[0],[1/m]]) 

    #Check controllability 
    print("Rank of controllability-matrix:", np.linalg.matrix_rank(control.ctrb(A,B)))

    return A,B

"""
Initialization of Cartpole system.
Given cartpole parameters, returns A and B matrices for cartpole in state-space form.
"""
def Cartpole_init(m = 1,M = 5,L = 2,g = -10,d = 1,s = 1):

    #A-matrix, defines dynamics of the DS
    A = np.array([[0,1,0,0],
              [0,(-d/M),(-m*g/M),0],
              [0,0,0,1],
              [0,(-s*d/(M*L)),(-s*(m+M)*g/(M*L)),0]])

    #B-matrix, defines the influence of the force on the system
    B = np.array([[0],[(1/M)],[0],[(s*1/(M*L))]])

    #Check controllability 
    print("Rank of controllability-matrix:", np.linalg.matrix_rank(control.ctrb(A,B)))

    return A,B

"""
Initialization of LQR gain matrix.
Given A and B matrices, as well as LQR parameters Q and R, returns LQR gain matrix Kc. 
"""
def Control_init(A,B,Q,R):

    #LQR gain matrix calculation
    Kc,_,_ = control.lqr(A,B,Q,R)

    return Kc

"""
Initialization of Kalman filter gain matrix.
Given A and C matrices, as well as covariance parameters for disturbance and noise, returns Kalman filter gain matrix Kf.
"""
def Kalman_init(A,C,Vn_cov=0.001,Vd_cov=0.001):

    #Covariance matrices
    Vd = Vd_cov*np.identity(len(A))  # disturbance covariance
    Vn = Vn_cov*np.identity(len(A))    # noise covariance

    #Kalman filter gain matrix calculation
    Kf_t,_,_=control.lqr(np.transpose(A),np.transpose(C),Vd,Vn)
    Kf=np.transpose(Kf_t)
    return Kf

"""
Initialization of state matrix X.
Given the starting state x0 and FE parameter Nt, returns X, a zero-matrix which keeps track of the system state.
"""
def X_init(x0,Nt):
    #Initialization of 'real system'
    X=np.zeros([len(x0),Nt+1])
    X[:,0]=x0

    return X

"""
Initialization of Kalman Filter SCN.
Given the SCN and FE parameters, returns SCN states with connections.
"""
def KfSCN_init(K,Nt,A,B,C,Kf,N=100,lam=0.1,bounding_box_factor=10,zero_init=True,x0=None,seed=0):

    np.random.seed(seed)

    D=np.random.randn(K,N) # N x K - Weights associated to each neuron
    D=D/np.linalg.norm(D,axis=0) #normalize
    D = D / bounding_box_factor # avoid too big discontinuities
    T = np.diag(D.T@D)/2

    # Initialize Voltage, spikes, rate
    V = np.zeros([N,Nt+1])
    s = np.zeros([N,Nt+1])
    r = np.zeros([N,Nt+1])

    # Set initial conditions
    if not zero_init:
        r[:,0] = np.array(np.linalg.pinv(D)@x0) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(x0-D@r[:,0])

    # Network connections:
    # - fast
    O_f = D.T @ D
    # - slow
    O_s = D.T @ (lam*np.identity(K) + A) @ D
    # - external input
    F_i = D.T @ B
    # - rec. kalman
    O_k = -D.T @ Kf @ C @ D
    # - ff kalman
    F_k = D.T @ Kf

    return D,T,V,s,r,O_f,O_s,F_i,O_k,F_k

"""
Initialization of SCN Controller.
Given the SCN and FE parameters, returns SCN states with connections.
"""
def ControllerSCN_init(K,Nt,A,B,C,Kf,Kc,N=100,lam=0.1,bounding_box_factor=10,zero_init=True,x0=None,seed=0):

    np.random.seed(seed)

    D=np.random.randn(K,N) # N x K - Weights associated to each neuron
    D=D/np.linalg.norm(D,axis=0) #normalize
    D = D / bounding_box_factor # avoid too big discontinuities
    T = np.diag(D.T@D)/2

    # Initialize Voltage, spikes, rate
    V = np.zeros([N,Nt+1])
    s = np.zeros([N,Nt+1])
    r = np.zeros([N,Nt+1])

    # Set initial conditions
    if not zero_init:
        r[:,0] = np.array(np.linalg.pinv(D)@x0) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(x0-D@r[:,0])

    #We require an index for the weights, as the connections are only relevant for the first K/2 weights (the rest are for encoding the target state)
    i=int(K/2)

    # Network connections:
    # - fast
    O_f = D[:-i].T @ D[:-i]
    # - slow
    O_s = D[:-i].T @ (lam*np.identity(i) + A) @ D[:-i]
    # - rec. control
    O_c = -D[:-i].T @ B @ Kc @ D[:-i]
    # - ff. control
    F_c = D[:-i].T @ B @ Kc
    # - rec. kalman
    O_k = D[:-i].T @ Kf @ C @ D[:-i]
    # - ff kalman
    F_k = -D[:-i].T @ Kf

    return D,T,V,s,r,O_f,O_s,O_c,F_c,O_k,F_k

"""
Initialization of the Kf loop.
Given parameters, returns other matrices which are of importance in the estimation loop, such as the observation matrix Y, noise matrices, control matrix U. 
Also returns X_hat and X_hat_fe, the matrices which keep track of the Kf estimation for both the SCN and the idealized KF respectively.
"""
def KfLoop_init(X,A,B,C,x0,Nt,Vd_cov,Vn_cov):

    U = np.zeros([B.shape[1],Nt+1])
    Y = np.zeros([C.shape[0],Nt+1])
    X_hat = np.zeros([len(x0),Nt+1])
    X_hat_fe = np.zeros([len(x0),Nt+1])
    
    Vd = Vd_cov*np.identity(len(A))
    Vn = Vn_cov*np.identity(len(A))

    uDIST = np.random.multivariate_normal(np.zeros(len(A)),Vd,Nt+1).T
    uNOISE = np.random.multivariate_normal(np.zeros(len(A)),Vn,Nt+1).T

    Y[:,0] = C@X[:,0] + uNOISE[:,0]

    return U,Y,X_hat,X_hat_fe,uDIST,uNOISE

"""
Initialization of the Control loop.
Given parameters, returns other matrices which are of importance in the control loop, such as the observation matrix Y, noise matrices, control matrix U. 
Also returns X_hat and X_hat_fe, the matrices which keep track of the controller estimation for both the SCN and the idealized controller respectively.
"""
def ControlLoop_init(X,X_2,error_scn,error_ideal,x_des,dt,A,B,C,x0,Nt,Vd_cov,Vn_cov):

    U = np.zeros([B.shape[1],Nt+1])
    Y = np.zeros([C.shape[0],Nt+1])
    
    U_2 = np.zeros([B.shape[1],Nt+1])
    Y_2 = np.zeros([C.shape[0],Nt+1])
    
    X_hat = np.zeros([len(x0),Nt+1])
    X_hat_fe = np.zeros([len(x0),Nt+1])
    
    Vd = Vd_cov*np.identity(len(A))
    Vn = Vn_cov*np.identity(len(A))

    uDIST = np.random.multivariate_normal(np.zeros(len(A)),Vd,Nt+1).T
    uNOISE = np.random.multivariate_normal(np.zeros(len(A)),Vn,Nt+1).T

    Y[:,0] = C@X[:,0] + uNOISE[:,0]
    Y_2[:,0] = C@X_2[:,0] + uNOISE[:,0]
    
    Dx=np.gradient(x_des,axis=1)/dt
    
    error_scn[:,0] = np.abs(X[:,0]-x_des[:,0])
    error_ideal[:,0] = np.abs(X_2[:,0]-x_des[:,0])

    return U,Y,U_2,Y_2,X_hat,X_hat_fe,uDIST,uNOISE,Dx,error_scn,error_ideal


import numpy as np

"""
Function for running a single step of the SCN Kalman filter network.
"""
def run_KfSCN_step(y,u,r,s,v,D,T,lam,O_f,O_s,F_i,O_k,F_k,C,t,dt,sigma):

    # Calculating the voltages at time t+1
    dvdt = -lam * v - O_f @ s + O_s @ r + F_i @ u + (O_k @ r + F_k @ C @ y)
    v_next = v + dvdt*dt + np.sqrt(dt)*sigma*np.random.randn(len(dvdt))

    # check if there are neurons whose voltage is above threshold
    above = np.where(v_next > T)[0]

    # introduce a control to let only one neuron fire at the time
    s_next=np.zeros(s.shape)
    if len(above):
        s_next[np.argmax(v_next)] = 1/dt

    # update rate
    drdt = s_next - lam*r
    r_next = r + drdt*dt
    
    return r_next, s_next, v_next

"""
Function for running a single step of the SCN controller.
"""
def run_SCNcontrol_step(y,x_des,Dx,r,s,v,D,T,lam,Kc,O_f,O_s,O_c,F_c,O_k,F_k,B,C,t,dt,sigma):
    
    #We require an index for the weights, as the connections are only relevant for the first B weights (the rest are for encoding the target state)
    i=len(B)
    
    u_next = -Kc @ (D[:-i] @ r - D[i:] @ r)

    # Calculating the voltages at time t+1
    dvdt = -lam * v - O_f @ s + O_s @ r + (O_c @ r + F_c @ D[i:] @ r) - (O_k @ r + F_k @ C @ y)
    dvdt = dvdt + (D[i:].T @ ((lam*x_des)+Dx)) - (D[i:].T @ D[i:] @ s)
    v_next = v + dvdt*dt + np.sqrt(dt)*sigma*np.random.randn(len(dvdt))

    # check if there are neurons whose voltage is above threshold
    above = np.where(v_next > T)[0]

    # introduce a control to let only one neuron fire at the time
    s_next=np.zeros(s.shape)
    if len(above):
        s_next[np.argmax(v_next)] = 1/dt

    # update rate
    drdt = s_next - lam*r
    r_next = r + drdt*dt
    
    return r_next, s_next, v_next, u_next

"""
Function for running a single step of the idealized Kalman filter.
"""
def run_Kfidealized_step(x_hat,A,B,u,Kf,y,C,dt):

    dxdt = A@x_hat + B@u + Kf@(y-(C@x_hat))
    x_next = x_hat + dxdt*dt
    
    return x_next

"""
Function for running a single step of a linearized Dynamical System (DS).
"""
def run_DSlinearized_step(x,A,B,u,dist,dt):
    
    dxdt = A@x + B@u 
    x_next = x + dxdt*dt + np.sqrt(dt)*dist
    
    return x_next

"""
Function for running a single step of a (non-linear) simulated Cartpole Dynamical System.
"""
def run_Cartpolereal_step(x,u,dist,m,M,L,g,d,dt):
    Sy=np.sin(x[2])
    Cy=np.cos(x[2])

    D = m*L*L*(M+m*(1-Cy**2))
    
    dy_1 = x[1]
    dy_2 = (1/D)*(-m**2*L**2*g*Cy*Sy + m*L**2*(m*L*x[3]**2*Sy - d*x[1])) + m*L*L*(1/D)*u
    dy_3 = x[3]
    dy_4 = (1/D)*((m+M)*m*g*L*Sy - m*L*Cy*(m*L*x[3]**2*Sy - d*x[1])) - m*L*Cy*(1/D)*u
    
    dxdt=np.array([dy_1, dy_2, dy_3, dy_4])

    x_next=x+dxdt*dt + np.sqrt(dt)*dist
    
    return x_next


import numpy as np #Numpy for matrix calculations
import matplotlib.pyplot as plt #Matplotlib for plotting
#import initialization, simulation #Helper functions for initialization and simulation are located in these two Python files. Please see the files themselves for more details.

#Forward Euler parameters
time = 50 #Total simulation time in seconds
dt = 0.001 #Length of a single timestep

#Spring-Mass-Damper System parameters
m = 3 #Mass (in kg)
k = 5 #Spring constant (in N/m)
c = 0.5 #Constant of proportionality (dampening, in Ns/m = kg/s)
x0 = np.array([5, 0]) #Initial state of the SMD system.

#Other system parameters
C = np.array([[1,0],
              [0,0]]) #Initialization of the C matrix (because y=Cx+noise)
Vn_cov = 0.001 #Sensor noise covariance (y=Cx+noise)
Vd_cov = 0.001 #Disturbance noise covariance (noise on the SMD)

#SCN Estimator parameters
network_size = 20 #The number of neurons in the SCN
signal_dimensions = 2 #The dimensions of the signal, K (is equal to the size of x0, but can be set manually)
lam = 0.1 #The leakage constant of the network, lambda
Vv_sigma = 0.000001 #Voltage noise sigma; noise on the voltage

#Forward Euler simulation
times,Nt = FE_init(time,dt) #times is a list of timesteps which we will loop over, Nt is the total number of timesteps (length of times)

#SMD System A and B matrices
A,B = SMD_init(m,k,c) #A and B are the system matrix and input matrix in state-space representation (according to Ax+Bu)

#Initialization of the Kalman filter gain matrix to be used inside of the SCN estimator and idealized Kalman filter
Kf = Kalman_init(A,C,Vn_cov,Vd_cov) #From the A and C matrices and noise covariances, we can calculate the Kalman filter gain matrix

#Initialization of the state-matrix, containing the states of the simulated SMD system over time
X = X_init(x0,Nt) #Requires x0 as the first state of the simulated SMD system, and Nt for the matrix dimensions

#Initializaton of the SCN estimator, given parameters, we calculate D, T, V, s, r and all of the connectivity
D,T,V,s,r,O_f,O_s,F_i,O_k,F_k = KfSCN_init(signal_dimensions,Nt,A,B,C,Kf,network_size,lam)

#Initialization of other matrices used in simulation, U, Y, X_hat (state matrix of SCN estimator), X_hat_fe (state matrix of idealized Kalman filter), uDIST and uNOISE (noise matrices)
U,Y,X_hat,X_hat_fe,uDIST,uNOISE = KfLoop_init(X,A,B,C,x0,Nt,Vd_cov,Vn_cov)

def run_simulation(Nt,X,A,B,U,uDIST,dt,Y,C,uNOISE,r,s,V,D,T,lam,O_f,O_s,F_i,O_k,F_k,Vv_sigma,X_hat,X_hat_fe,Kf):
    #Looping over the entire range of Nt, we have all the timesteps in our simulation
    for t in range(Nt):
        #First, simulate one step of the simulated SMD system
        X[:,t+1] = run_DSlinearized_step(X[:,t],A,B,U[:,t],uDIST[:,t],dt)

        #Our Kalman filters only have access to Y, which is the partially observable state plus noise
        Y[:,t+1] = C@X[:,t+1] + uNOISE[:,t+1]

        #Simulate a single step of the SCN Kalman filter
        r[:,t+1],s[:,t+1],V[:,t+1] = run_KfSCN_step(Y[:,t],U[:,t],r[:,t],s[:,t],V[:,t],D,T,lam,O_f,O_s,F_i,O_k,F_k,C,t,dt,Vv_sigma)
        
        #Calculate the state estimated by our SCN Kalman filter by decoding the internal firing rates
        X_hat[:,t+1] = D@r[:,t+1]
        
        #Run a step of the idealized Kalman filter which we compare the SCN to
        X_hat_fe[:,t+1] = run_Kfidealized_step(X_hat_fe[:,t],A,B,U[:,t],Kf,Y[:,t],C,dt)

        #We set U to zero, since we are using no outside input in this estimation plot
        U[0,t+1] = 0
    
    return X_hat,X_hat_fe,X,Y,s

X_hat,X_hat_fe,X,Y,s = run_simulation(Nt,X,A,B,U,uDIST,dt,Y,C,uNOISE,r,s,V,D,T,lam,O_f,O_s,F_i,O_k,F_k,Vv_sigma,X_hat,X_hat_fe,Kf)


fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(12)
fig, axs = plt.subplots(3,1, sharex=True, squeeze=True, gridspec_kw = {'hspace':0.1,'height_ratios':[1,1,1]})
legend_fontsize=12

legend=[]
axs[0].plot(np.arange(0,time+dt,dt),Y[0],color='#0070C0',alpha=0.7)
legend.append("Observation")
#axs[0].plot(np.arange(0,Time+dt,dt),X_hat_fe[0],color='#0000FF',linewidth=3)
#legend.append("x̂$_{"+str(1)+"}$_FE")
axs[0].plot(np.arange(0,time+dt,dt),X[0],color='#E3000B')
legend.append("SMD System")
axs[0].plot(np.arange(0,time+dt,dt),X_hat[0],color='#00B050')
legend.append("SCN Estimator")

axs[1].plot(np.arange(0,time+dt,dt),X[1],color='#E3000B')
axs[1].plot(np.arange(0,time+dt,dt),X_hat[1],color='#00B050')
fig.legend(legend,fontsize=legend_fontsize,loc='upper right',bbox_to_anchor=(0.90,0.74))
    
axs[0].set_ylabel('$x$ $(m)$',fontsize = 12)
axs[1].set_ylabel('$v$ $(m/s)$',fontsize = 12)

#We use a scatterplot for the spike trains:
axs[2].scatter(np.nonzero(s)[1]/1000,np.nonzero(s)[0],marker=".",s=0.1,color='black')
axs[2].set_xlabel('time ($s$)',fontsize = 12)
axs[2].set_ylabel('neuron nr.',fontsize = 12)

axins = axs[0].inset_axes([0.5, 0.85, 0.4, 0.8])
range_plot_x,range_plot_y=40000,45000
axins.plot(np.arange(0,time+dt,dt)[range_plot_x:range_plot_y],Y[0,range_plot_x:range_plot_y],color='#0070C0',alpha=0.7)
axins.plot(np.arange(0,time+dt,dt)[range_plot_x:range_plot_y],X[0,range_plot_x:range_plot_y],color='#00B050')
axins.plot(np.arange(0,time+dt,dt)[range_plot_x:range_plot_y],X_hat[0,range_plot_x:range_plot_y],color='#E3000B')
axins.set_xticklabels([])
axins.set_yticklabels([])

axs[0].indicate_inset_zoom(axins, edgecolor="black")

plt.show()