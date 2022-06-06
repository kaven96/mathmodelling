int: N = 20;
set of int: ROW = 1..N;
set of int: COL = 1..N;

array[ROW, COL] of var 0..1: x;

constraint
  forall([sum([x[i,j] | j in COL]) = 1 | i in ROW]);
  
constraint
  forall(j in COL)(
    sum(i in ROW)(x[i,j]) = 1
  );

%возможное ограничение снятия симметрии: sum(j in 1..N div 2)(x[1,j]) = 1
  
constraint
  forall(c in 2-N..N-2)(
    sum(i in max(1, 1-c)..min(N, N-c))(x[i,     c+i]) <= 1
    /\
    sum(i in max(1, 1+c)..min(N, N+c))(x[i, N+1+c-i]) <= 1
  );
  
output [
  join("\n", [
    join("  ", [if fix(x[r, c]) = 1 then "*" else "-" endif | c in COL])
    | r in ROW
  ])
];