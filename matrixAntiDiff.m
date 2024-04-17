function [A,A2] = matrixAntiDiff(n,dx)
    %c for antidifferentiations
    c = ones(n, 1)*dx/2;
    
    % Construct trapezoidal antidifferentiation operator - n-1 x n
    A = tril(ones(n,n))*dx;
    A(:,1) = c(1:end);
    A = A - diag(c,0);

    %Construct second-order trapezoidal antidifferentiation operator - n-1 x n
    A2 = A*A;

    %Truncate matrices
    A = A(2:end,:);
    A2 = A2(2:end,:);
end

%% Copyright notice
% Copyright 2023 Luis Viornery.

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