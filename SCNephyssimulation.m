function spikes = SCNephyssimulation()

    N = 2^10;  % number of neurons
    M = 14^2;  % number of pixels (M1 ipRGC cells) - must be a perfect square
    
    % electrophysiology parameters
    param = defaultparameters();
    
    % more physical parameters
    gaba_conn_prob    = 0.01;     % the density of connections within the SCN
    num_pixel_to_scn  = 1;        % the number of connections from each pixel to the SCN
    exgabafrac        = 0.2;      % fraction of SCN neurons that have an excitatory response to gaba
    CTstd             = 2;        % hr, standard deviation of circadian times
    Egabaex           = -40;      % mV, excitatory gaba reversal potential
    Egabain           = -80;      % mV, inhibitory gaba reversal potential
    inputtype         = 0;        % 0 for full field, 1/-1 for vertical/horizontal bars of light
    
    % numerical parameters
    tf     = 15000;  % ms, final time of the simulation
    dt     = 0.1;    % ms, fixed timestep
    pertV  = 20;     % mV, +/- range of uniform perturbation to initial membrane voltages
    
    % generate connectivities
    Cgaba = ceil(sprand(N,N,gaba_conn_prob)); % on average N^2*gaba_conn_prob non-zeros
    m_in = repmat( 1:M, num_pixel_to_scn , 1 );
    n_out = NaN(num_pixel_to_scn,M);
    for m=1:M
        n_out(:,m) = randperm(N,num_pixel_to_scn)'; % no replacement with randperm
    end
    Cin = sparse(n_out(:),m_in(:),ones(size(n_out)),N,M);
        
    % setup the SCN population
    x = defaultic();
    x = structfun( @(in) repmat(in,N,1), x, 'UniformOutput', false ); % expand initial condition
    x.v = x.v + (rand(N,1)*2-1)*pertV; % perturb the initial voltages to desynchronize the population
    param = structfun( @(in) repmat(in,N,1), param, 'UniformOutput', false ); % expand parameters 
    
    param.spikethres = -20; % mV, minimum peak value of voltage to record a spike
    param.dtspikemin =   5; % ms, minimum time between peaks to record a spike
    % perturb CT over the population to introduce heterogeneity 
    % and to account for multiple experimental times
    param.CT = param.CT + CTstd*randn(N,1);
    param = setcircadiantime(param);
    
    % select GABA reversal potentials
    param.Egaba = Egabain*ones(N,1);
    param.Egaba( datasample( 1:N, round( exgabafrac*N ), 'Replace',false ) ) = Egabaex;
    
    % timestepping setup
    numsteps = ceil(tf/dt);
    step = 0;
            
    % setup for spike detection
    spikedata.tlast = NaN(N,1);
    spikedata.spikes = zeros(0,2);
    spikedata.v1 = NaN(N,1);
    spikedata.v2 = NaN(N,1);
    spikedata.thres      = -20; % mV - minimum peak value of voltage to record a spike
    spikedata.dtspikemin =   5; % ms - minimum time between peaks to record a spike

    while step < numsteps
        
        t = step*dt;
        step = step + 1;
                
        % update the state variables
        [x,spikedata] = onestepupdates(t,x,param,dt,Cgaba,Cin,spikedata,inputtype);
                
    end % of timestepping loop

    spikes = spikedata.spikes;
    
    % raster plot
    plot(spikes(:,2)/1000,spikes(:,1),'ok')
    xlabel('time [s]')
    ylabel('neuron number')
    axis([0 tf/1000 0 N+1])
end

function [x,spikedata] = onestepupdates(t,x,param,dt,Cgaba,Cin,spikedata,inputtype)
% updates the scn electrophysiology states variables from t to t + dt
% t - time at which voltage is defined, all other variables are defined at t-dt/2
% x - structure of current state variables
% param - structure of parameters
% dt - stepsize
% Cgaba - connection matrix for the gaba network
% Cin - connection matrix for the input
% the gating variables, calcium variables, and v 'leapfrog'
% start: v is defined at t, gates, and calcium at t-dt/2
% end: v at t+dt, gates and calcium at t+dt/2
    
    % evaluate expressions for gating dynamics
    % assuming v and Cas are fixed
    qinf = [1.0./(1.0+exp(-(x.v+35.2)/8.1)), 1.0./(1.0+exp((x.v+62.0)/2.0)), 1.0./(1.0+exp((x.v-14.0)/(-17.0))).^0.25, 1.0./(1.0+exp(-(x.v+36.0)/5.1)), 1.0./(1.0+exp(-(x.v+21.6)/6.7)), 1.0./(1.0+exp((x.v+260.0)/65.0)), 1e7*x.ca(:,1).^2./(1e7*x.ca(:,1).^2+5.6)];
    tauq = [exp(-(x.v+286.0)/160.0), 0.51+exp(-(x.v+26.6)/7.1), exp(-(x.v-67.0)/68.0), param.taurl, param.taurnl, exp(-(x.v-444.0)/220.0), 500.0./(1e7*x.ca(:,1).^2+5.6)];
        
	% update gating variables, except s
    qold = x.q;
    x.q(:,1:6) = ( 2.0*dt*qinf(:,1:6) + (2.0*tauq(:,1:6)-dt).*x.q(:,1:6) ) ./ (2.0*tauq(:,1:6)+dt);
    
	% update Cas (solve a quadratic equation)
    % before s and cac (since they depend on Cas)
    casold = x.ca(:,1);
    
    expr = -param.kca(:,1).*(param.gcal.*qold(:,4).*param.K1./(param.K2+x.ca(:,1))+param.gcanl.*qold(:,5).*qold(:,6)).*(x.v-param.Eca) - x.ca(:,1)./param.tauca(:,1) + param.bca(:,1);
    B = ( param.K2 - x.ca(:,1) - dt/2*expr + dt/2*param.kca(:,1).*param.gcanl.*x.q(:,5).*x.q(:,6).*(x.v-param.Eca) + dt/2./param.tauca(:,1).*param.K2 - param.bca(:,1)*dt/2 ) ./ (1+dt/2./param.tauca(:,1));
    C = (-param.K2.*x.ca(:,1) - dt/2*param.K2.*expr + dt/2*param.kca(:,1).*param.gcal.*x.q(:,4).*param.K1.*(x.v-param.Eca)+dt/2*param.kca(:,1).*param.gcanl.*x.q(:,5).*x.q(:,6).*(x.v-param.Eca).*param.K2 - dt/2*param.bca(:,1).*param.K2 ) ./ (1+dt/2./param.tauca(:,1));
    x.ca(:,1) = (sqrt(B.^2-4*C)-B)/2;
    
    % update s
    ca2 = 1e7*x.ca(:,1).^2;
    sinf = 1./(1+5.6./ca2);
	taus = 500.0./(ca2+5.6);
	x.q(:,7)= ( x.q(:,7).*(1.0-dt./(2.0*tauq(:,7)))+dt/2.0*(qinf(:,7)./tauq(:,7)+sinf./taus) )./(1.0+dt./(2.0*taus));
    
    % update Cac
    x.ca(:,2) = (   x.ca(:,2).*(1.0-dt./(2.0*param.tauca(:,2))) + param.bca(:,2)*dt - dt*param.kca(:,2)/2.0.*( param.gcal.*x.q(:,4).*(param.K1./(param.K2+x.ca(:,1)))+param.gcanl.*x.q(:,5).*x.q(:,6)+param.gcal.*qold(:,4).*(param.K1./(param.K2+casold))+param.gcanl.*qold(:,5).*qold(:,6) ).*(x.v-param.Eca)   )./(1.0+dt./(2.0*param.tauca(:,2)));

    % update the gaba gating variable
    T = param.Tmax ./ ( 1+exp(-(x.v-param.Vt)./param.Kp) );
    R = param.ar.*T+param.ad;

    x.y = ( (2.0-dt*R).*x.y + 2*dt*param.ar.*T )./(2.0+dt*R);
    
    % compute the total conductance
    ggaba = param.ggabamax.*(Cgaba*x.y);
    
    % compute the total conductances
    gna = param.gna.*x.q(:,1).*x.q(:,1).*x.q(:,1).*x.q(:,2);
    gk = param.gk.*x.q(:,3).*x.q(:,3).*x.q(:,3).*x.q(:,3);
    gkca = param.gkca.*x.q(:,7).*x.q(:,7);
    gcal = param.gcal.*x.q(:,4).*(param.K1./(param.K2+x.ca(:,1)));
    gcanl = param.gcanl.*x.q(:,5).*x.q(:,6);

	% update the voltage
    G   = gna + gk + gcal + gcanl + gkca + param.gkleak + param.gnaleak + ggaba;
    I0  = param.Ena.*( gna+param.gnaleak ) + param.Ek.*( gk+gkca+param.gkleak ) + param.Eca.*( gcal+gcanl ) + param.Egaba.*ggaba;
    Iin = computeinput( t + dt/2, Cin, param, inputtype );
    x.v = ( dt*(param.Iapp+Iin+I0) + (param.Cm-dt/2.0*G).*x.v )./( param.Cm+dt/2.0*G );
    
    % detect spike
    a = ( x.v-2*spikedata.v1+spikedata.v2)/(2*dt*dt);
    b = (-x.v+4*spikedata.v1-3*spikedata.v2)/(2*dt);
    text = t-dt-b./a/2;
    val = a.*(text-t+dt).*(text-t+dt) + b.*(text-t+dt) + spikedata.v2;
    spikedata.v2 = spikedata.v1;
    spikedata.v1 = x.v;
    
    ind = 1:numel(x.v);
    ind = ind( t-dt < text & text < t+dt & ( isnan(spikedata.tlast) | text>(spikedata.tlast+spikedata.dtspikemin) ) & val > spikedata.thres );
    for ii = ind
        spikedata.spikes = [spikedata.spikes;ii,text(ii)];
        spikedata.tlast(ii) = text(ii);
    end
    
end

function Iin = computeinput( t, Cin, param, inputtype )
% returns the current input to the SCN neurons at time t

    [N,M] = size(Cin);
    sqM = sqrt(M);
    input = zeros(sqM);
    
    gridpoints = (0.5:sqM)/sqM;
    [X,Y] = meshgrid( gridpoints, gridpoints ); % the centers of the pixels
    
    if inputtype == 0 % full field flash
        
        % 5 sec off, 5 sec on, 5 sec off
        Toff1 = 5000; Ton  = 5000; Toff2 = 5000;
        T = Ton + Toff1 + Toff2;
        tm = (t - floor(t/T)*T);
        if ( Toff1 < tm && tm < Toff1+Ton ) 
            input = 1+input;
        end        
        
    elseif inputtype == 1 || inputtype == -1 % bar
        
        P = 250; % duration of the bar presentation
        toppixel = [46,42,2,54,34,3,17,69,13,58,19,55,67,61,36,29,63,35,18,65,20,68,31,4,51,12,59,39,44,7,16,48,21,23,49,47,53,57,25,22,5,11,70,43,60,14,27,1,6,37,32,41,30,33,0,9,56,62,66,50,15,40,52,8,38,24,45,64,26,28,10]/76;
        tind = 1+mod(floor(t/(2*P)),length(toppixel));
        wid = ceil(6/76*sqM); % width of the bar
        ind = 1+floor(toppixel(tind)*sqM);
        if mod(floor(t/P),2)==0 % on for 250ms, off for 250ms
            if inputtype == 1
                % vertical bar
                input( ind:(ind+wid-1) , : ) = 1;
            else
                % horizontal bar
                input( :, ind:(ind+wid-1) ) = 1;
            end
        end
    end
    
    % scale the input on [0,1] to the units of current
    Iin = (param.iF-param.iA) .* ( Cin*input(:) ) + param.iA;
    
end

function param = defaultparameters()
% default ephys parameters

    % physical parameters
    param.Cm        = 5.7;                  % pF                    membrane capacitance per unit area
    param.gna       = 229;                  % nS                    Na+ conductance per unit area
    param.gnaleak   = 0.0576;               % nS                    Na+ leak conductance per unit area
    param.gk        = 3;                    % nS                    K+ conductance per unit area
    param.gkleak    = 0.0333;               % nS                    K+ leak conductance per unit area
    param.gcal      = 6;                    % nS                    Ca++ conductance per unit area for L-type channel
    param.gcanl     = 20;                   % nS                    Ca++ conductance per unit area for non L-type channel
    param.gkca      = 100;                  % nS                    Ca++ activated K+ current
    param.Ena       = 45;                   % mV                    equilibrium potential for Na+
    param.Ek        = -97;                  % mV                    equilibrium potential for K+
    param.Eca       = 54;                   % mV                    equilibrium potential for K+
    param.K1        = 3.93*1e-5;            % mM                    parameter for the value of fL
    param.K2        = 6.55*1e-4;            % mM                    parameter for the value of fL
    param.kca       = [1.65e-4, 8.59e-9];   % mM / fC               calcium current to concentration conversion factor
    param.tauca     = [0.1,1.75e3];         % ms                    calcium clearance time constant
    param.bca       = [5.425e-4, 3.1e-8];   % mM / ms 
    param.Iapp      = 0;                    % pA                    constant applied current into the cell
    param.ar        = 5;                    % 1 / mM / ms           activation rate of the gaba synapse
    param.ad        = 0.18;                 % 1 / ms                de-activation rate of the gaba synapse
    param.Tmax      = 1;                    % mM                    maximum neurotransmitter in the gaba synapse
    param.Vt        = -20;                  % mV                    neurontransmitter threshold
    param.Kp        = 3;                    % mV                    neurontransmitter activation rate
    param.taurl     = 3.1;                  % ms                    time constant for the rl gate
    param.taurnl    = 3.1;                  % ms                    time constant for the rnl gate
    param.ggabamax  = 0.6;                  % nS                    max conductance of gaba synaptic inputs
    param.Egaba     = -80;                  % mV                    gaba synapse reversal potential
    param.iF        = 10;                   % pA                    maximum input current (full)
    param.iA        =  0;                   % pA                    background input current (ambient)
    param.CT        = 14.6;                 % hr                    circadian time, the mean of the times of the experiments
    
    param = setcircadiantime(param);

end

function param = setcircadiantime(param)
% sets the ephys parameters gkca and gkleak

    % pre-computed values (CT,gkca,gkleak)
    X=[0.00000000 87.64196088 0.08650703; 0.28717949 69.46008967 0.06814150; 0.58021978 53.48868503 0.05200877;
    0.87326007 40.59755923 0.03898743; 1.16630037 30.67790598 0.02896758; 1.45934066 23.29286765 0.02150795;
    1.75238095 17.90764710 0.01606833; 2.04542125 14.02490751 0.01214637; 2.33846154 11.23854569 0.00933186;
    2.63150183 9.24010177 0.00731323; 2.92454212 7.80501806 0.00586365; 3.21758242 6.77392888 0.00482215;
    3.51062271 6.03555354 0.00407632; 3.80366300 5.51319319 0.00354868; 4.09670330 5.15486316 0.00318673;
    4.38974359 4.92645389 0.00295601; 4.68278388 4.80724370 0.00283560; 4.97582418 4.78722721 0.00281538;
    5.26886447 4.86591347 0.00289486; 5.56190476 5.05244686 0.00308328; 5.85494505 5.36709760 0.00340111;
    6.14798535 5.84437518 0.00388321; 6.44102564 6.53823867 0.00458408; 6.73406593 7.53009911 0.00558596;
    7.02710623 8.94044754 0.00701055; 7.32014652 10.94475317 0.00903510; 7.61318681 13.79319424 0.01191232;
    7.90622711 17.83070360 0.01599061; 8.19926740 23.50700497 0.02172425; 8.49230769 31.35458278 0.02965109;
    8.78534799 41.89946789 0.04030249; 9.07838828 55.47218441 0.05401231; 9.37142857 71.93822724 0.07064467;
    9.66446886 90.48177832 0.08937553; 9.95750916 109.66201402 0.10874951; 10.25054945 127.83742991 0.12710852;
    10.54358974 143.73285326 0.14316450; 10.83663004 156.76131430 0.15632456; 11.12967033 166.95754344 0.16662378;
    11.42271062 174.70983174 0.17445438; 11.71575092 180.51502490 0.18031821; 12.00879121 184.83887735 0.18468573;
    12.30183150 188.06342575 0.18794285; 12.59487179 190.48142125 0.19038527; 12.88791209 192.30934315 0.19223166;
    13.18095238 193.70446539 0.19364087; 13.47399267 194.78027469 0.19472755; 13.76703297 195.61862848 0.19557437;
    14.06007326 196.27878962 0.19624120; 14.35311355 196.80392679 0.19677164; 14.64615385 197.22572579 0.19719770;
    14.93919414 197.56764003 0.19754307; 15.23223443 197.84717946 0.19782543; 15.52527473 198.07753093 0.19805811;
    15.81831502 198.26870647 0.19825122; 16.11135531 198.42836231 0.19841249; 16.40439560 198.56238881 0.19854787;
    16.69743590 198.67532497 0.19866194; 16.99047619 198.77065581 0.19875824; 17.28351648 198.85100765 0.19883940;
    17.57655678 198.91824471 0.19890732; 17.86959707 198.97344542 0.19896308; 18.16263736 199.01667095 0.19900674;
    18.45567766 199.04618442 0.19903655; 18.74871795 199.05656634 0.19904704; 19.04175824 199.03467479 0.19902492;
    19.33479853 198.95936443 0.19894885; 19.62783883 198.80444823 0.19879237; 19.92087912 198.54153063 0.19852680;
    20.21391941 198.12142481 0.19810245; 20.50695971 197.47845528 0.19745299; 20.80000000 196.50094803 0.19646560;
    21.09304029 195.02428011 0.19497402; 21.38608059 192.80663103 0.19273397; 21.67912088 189.50552139 0.18939952;
    21.97216117 184.65992618 0.18450498; 22.26520147 177.69905590 0.17747379; 22.55824176 168.01510880 0.16769203;
    22.85128205 155.13927121 0.15468613; 23.14432234 139.01713565 0.13840115; 23.43736264 120.27176544 0.11946643;
    23.73040293 100.23838389 0.09923069;24.00000000 82.14208394 0.08095160];

    param.gkca = interp1( X(:,1),X(:,2), param.CT(:));
    param.gkleak = interp1( X(:,1),X(:,3), param.CT(:));
end

function x = defaultic()
% returns the default initial condition and a coefficient of variation
% over the population

    % v,q,ca on limit cycle
    % y at steady value
    x.v  = -81.4;
    x.q  = [0.0033, 0.0297, 0.2655, 0.0002, 0.0002, 0.0551, 0.0651]; % m, h, n, rl, rnl, fnl, s
    x.ca = [5.485e-5, 1e-4]; % cas, cac
    x.y  = 3.59e-8; % synapse variable
    
end
