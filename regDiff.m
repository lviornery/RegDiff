% function [u,v] = regDiff(data, alpha, alpha2, dx, u0, maxiter, deltacosttol, deltanormtol, ep, plotflag, diagflag)
% Luis Viornery (lviornery@cmu.edu), November 27, 2023, modified from code
% by Rick Chartrand (rickc@lanl.gov)
% Please cite <TBD>
%
% Inputs:  (First three required; omitting the final N parameters for N < 7
%           or passing in [] results in default values being used.) 
%       data        Vector of data to be differentiated. Required.
%
%       alpha       Regularization parameter.  This is the main parameter
%                   to fiddle with.  Start by varying by orders of
%                   magnitude until reasonable results are obtained.  A
%                   value to the nearest power of 10 is usally adequate.
%                   No default value. Required. Higher values increase
%                   regularization strength and improve conditioning.
%
%       alpha2      Regularization parameter for second-order derivative.
%                   No default value. Skips second-order differentiation
%                   if omitted.  Higher values increase regularization
%                   strength and improve conditioning.
%
%       dx          Data spacing. Default is 1.
%
%       u0          Initialization of the derivatives. Can be a 1x2 cell
%                   array containing nx1 vectors, an nx2 matrix, or an nx1
%                   vector. If a cell array, the first vector (if not
%                   empty) is used for the first derivative and the second 
%                   vector (if not empty) is used for the second
%                   derivative. If a matrix, the first column is used for 
%                   the first derivative and the second column is used for 
%                   the second derivative. If a vector, used for the first 
%                   derivative. Calculated using a central difference
%                   method by default. Although the solution is
%                   theoretically independent of the intialization, a poor
%                   choice can exacerbate conditioning issues when the
%                   linear system is solved.
%
%       maxiter     Maximum number of iterations to run the solver loop.
%                   Default is 100.
%
%       deltacosttol    Minimum relative change in the cost. Value below 
%                       this terminates solver. Default is 1e-6.
%
%       deltanormtol    Minimum relative norm of adjustment. Value below 
%                       this terminates solver. Default is 1e-6.
%
%       ep          Parameter for avoiding division by zero.  Default value
%                   is 1e-6.  Results should not be very sensitive to the
%                   value.  Larger values improve conditioning and
%                   therefore speed, while smaller values give more
%                   accurate results with sharper jumps.
%
%       plotflag    Flag whether to display plot at each iteration.
%                   Default is 0 (no).  Useful, but adds significant
%                   running time.
%
%       diagflag    Flag whether to display diagnostics at each
%                   iteration.  Default is 0 (no).  Useful for diagnosing
%                   preconditioning problems.  When tolerance is not met,
%                   an early iterate being best is more worrying than a
%                   large relative residual.
%                   
% Output:
%
%       u           Estimate of the regularized derivative of data.
%
%       v           Estimate of the regularized second derivative of data,
%                   calculated directly from the data

function [u,v,ucost,vcost] = regDiff(data, alpha, alpha2, dx, u0, maxiter, deltacosttol, deltanormtol, ep, plotflag, diagflag)
    % Make sure we have a column vector.
    data = data( : );
    % Get the data size.
    n = length( data );
    %normalize the data
    datamean = mean(data);
    datastdev = std(data);
    data = (data - datamean)/datastdev;

    if nargin < 4 || isempty(dx)
        dx = 1;
    end

    if nargin < 3 || isempty(alpha2)
        secondDeriv = false;
    else
        secondDeriv = true;
    end
    
    % Set up matrices
    [Diff,Diff2,D] = matrixDiff(n,1);
    DT = D';
    
    %construct antidifferentiation and data fidelity operators
    [A,A2] = matrixAntiDiff(n,1);
    DF = A';
    DFA = DF*A;
    DFA2 = DF*A2;
    
    %set defaults
    if nargin < 11 || isempty(diagflag)
        diagflag = 0;
    end
    if nargin < 10 || isempty(plotflag)
        plotflag = 0;
    end
    if nargin < 9 || isempty(ep)
        ep = 1e-8;
    end
    if nargin < 8 || isempty(deltanormtol)
        deltanormtol = 1e-6;
    end
    if nargin < 7 || isempty(deltacosttol)
        deltacosttol = 1e-6;
    end
    if nargin < 6 || isempty(maxiter)
        maxiter = 100000;
    end

    % Default initialization of derivatives is central difference.
    if nargin < 5 || isempty(u0)
        u = Diff*data;
        v = Diff2*data;
    elseif isa(u0,'cell')
        if length(u0) >= 1
            u = u0{1}*dx/datastdev;
        else
            u = Diff*data;
        end
        if length(u0) >= 2
            v = u0{2}*dx^2/datastdev;
        else
            v = Diff2*data;
        end
    else
        if size(u0,2) >= 1
            u = u0(:,1)*dx/datastdev;
        else
            u = Diff*data;
        end
        if size(u0,2) >= 2
            v = u0(:,2)*dx^2/datastdev;
        else
            v = Diff2*data;
        end
    end
    
    %First order differentiation
    ofst1 = data(1);
    datacomp = data(2:end) - ofst1;
    DFb = DF*datacomp;
    if diagflag
        ucost = norm(cumtrapz(u) - [0;datacomp],2) + alpha*norm(diff(u),1);
    else
        ucost = [];
    end
    prevcost = [];
    for i = 1:maxiter
        % Diagonal matrix of 1/|u'|
        En = spdiags(1./(sqrt(diff(u).^2 + ep)), 0, n-1, n-1);
        % Linearized diffusion matrix, also approximation of Hessian.
        Ln = DT * En * D; % nxn
        % Gradient of functional.
        gn = DFA*u - DFb + alpha * Ln * u;
        Hn = -(DFA + alpha*Ln);
        s = Hn\gn;
        u = u + s;
        costn = norm(cumtrapz(u) - [0;datacomp],2) + alpha*norm(diff(u),1);
        if diagflag
            fprintf('u, iteration %4d: cost %.3e, relative change = %.3e\n',...
                i,...
                costn,...
                norm(s) / norm(u));
            ucost = [ucost;costn];
        end
        % Display plot.
        if plotflag
            if secondDeriv
                subplot(2,1,1)
            end
            plot( u, 'ok' ), drawnow;
        end
        if norm(s) / norm(u) < deltanormtol
            if diagflag
                fprintf('u, delta norm tolerance reached \n')
            end
            break;
        end
        if ~isempty(prevcost) && (prevcost - costn) / prevcost < deltacosttol
            if diagflag
                fprintf('u, relative cost tolerance reached \n')
            end
            break;
        end
        prevcost = costn;
    end
    
    if secondDeriv%Second-order differentiation
        ofst2 = u(1);
        datacomp = data(2:end) - ofst1 - ofst2*(1:(n-1))';
        DFb = DF*datacomp;
        if diagflag
            vcost = norm(cumtrapz(cumtrapz(v)) - [0;datacomp],2) + alpha2*norm(diff(v),1);
        else
            vcost = [];
        end
        prevcost = [];
        for i = 1:maxiter
            % Diagonal matrix of 1/|v'|
            En = spdiags(1./(sqrt(diff(v).^2 + ep)), 0, n-1, n-1);
            % Linearized diffusion matrix, also approximation of Hessian.
            Ln = DT * En * D;
            % Gradient of functional.
            gn = DFA2*v - DFb + alpha2 * Ln * v;% Prepare to solve linear equation.
            Hn = -(DFA2 + alpha2*Ln);
            % Update solution.
            s = Hn\gn;
            v = v + s;
            costn = norm(cumtrapz(cumtrapz(v)) - [0;datacomp],2) + alpha2*norm(diff(v),1);
            if diagflag
                fprintf('v, iteration %4d: cost %.3e, relative change = %.3e\n',...
                    i,...
                    costn,...
                    norm(s) / norm(v));
                vcost = [vcost;costn];
            end
            % Display plot.
            if plotflag
                subplot(2,1,2)
                plot( v, 'ok' ), drawnow;
            end
            if norm(s) / norm(v) < deltanormtol
                if diagflag
                    fprintf('v, delta norm tolerance reached \n')
                end
                break;
            end
        if ~isempty(prevcost) && (prevcost - costn) / prevcost < deltacosttol
            if diagflag
                fprintf('v, relative cost tolerance reached \n')
            end
            break;
        end
        prevcost = costn;
        end
        v = v*datastdev/(dx^2);
    end

    u = u*datastdev/dx;
end


%% Copyright notice
% Copyright 2023 Luis Viornery.

%% Original Copyright notice:
% Copyright 2010. Los Alamos National Security, LLC. This material
% was produced under U.S. Government contract DE-AC52-06NA25396 for
% Los Alamos National Laboratory, which is operated by Los Alamos
% National Security, LLC, for the U.S. Department of Energy. The
% Government is granted for, itself and others acting on its
% behalf, a paid-up, nonexclusive, irrevocable worldwide license in
% this material to reproduce, prepare derivative works, and perform
% publicly and display publicly. Beginning five (5) years after
% (March 31, 2011) permission to assert copyright was obtained,
% subject to additional five-year worldwide renewals, the
% Government is granted for itself and others acting on its behalf
% a paid-up, nonexclusive, irrevocable worldwide license in this
% material to reproduce, prepare derivative works, distribute
% copies to the public, perform publicly and display publicly, and
% to permit others to do so. NEITHER THE UNITED STATES NOR THE
% UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
% SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
% EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
% RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
% ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
% REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
% RIGHTS. 

%% BSD License notice:
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met: 
% 
%      Redistributions of source code must retain the above
%      copyright notice, this list of conditions and the following
%      disclaimer.  
%      Redistributions in binary form must reproduce the above
%      copyright notice, this list of conditions and the following
%      disclaimer in the documentation and/or other materials
%      provided with the distribution. 
%      Neither the name of Los Alamos National Security nor the names of its
%      contributors may be used to endorse or promote products
%      derived from this software without specific prior written
%      permission. 
%      Neither the name of Luis Viornery nor the name of Carnegie Mellon 
%      University may be used to endorse or promote products derived from 
%      this software without specific prior written permission.
%  
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
% CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
% INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
% USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
% AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
% LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
% ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE. 