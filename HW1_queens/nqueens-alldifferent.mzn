% input problem size
int: N = 30;

% number of elements in rows and columns of table
% vectors of size N such as i-th element equal to i
set of int: ROW = 1..N;
set of int: COL = 1..N;

% vector of variables which holds queens positions
set of int: QUEEN = 1..N;

% declare array called r whith the same size as QUEEN vector
% each i-th element of r is equal to to the i-th element of QUEEN
array[QUEEN] of ROW: r = [i | i in QUEEN];

%declare empty vector called c with the same size as QUEEN vector
array[QUEEN] of var COL: c;

include "globals.mzn"; % idk why we need this, maybe to load standard libs

% constraint alldifferent(r);

% all elements in c must be different 'coz no any 2 queens can be in the same column
constraint alldifferent(c);

% if sums and differences of the cells indexes 
constraint alldifferent([r[q]+c[q] | q in QUEEN]);
constraint alldifferent([r[q]-c[q] | q in QUEEN]);

  
output [
  join("\n", [
    join("  ", [if exists(q in QUEEN)(fix(r[q])=i /\ fix(c[q])=j) then "*" else "-" endif | j in COL])
    | i in ROW
  ])
];