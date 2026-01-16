function [Diff,Diff2,D] = matrixDiff(n,dx)
%First central difference
c = ones(n, 1)/(2*dx);
Diff = spdiags([-c, c], [-1, 1], n, n);
Diff(1,1:3) = [-3, 4, -1]/(2*dx);
Diff(end,end-2:end) = [1, -4, 3]/(2*dx);

%Second cetnral difference
c = ones(n, 1)/dx^2;
Diff2 = spdiags([c, -2*c, c], [-1, 0, 1], n, n);
Diff2(1,1:4) = [2, -5, 4, -1]/(dx^2);
Diff2(end,end-3:end) = [-1, 4, -5, 2]/(dx^2);

%first order differencing matrix
c = ones(n,1)/dx;
D = spdiags([-c,c],[0,1],n-1,n);
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