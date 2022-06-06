int: N = 30;
set of int: ROW = 1..N;
set of int: COL = 1..N;
set of int: QUEEN = 1..N;

array[QUEEN] of var ROW: r;
array[QUEEN] of var COL: c;

constraint 
  forall(q1, q2 in QUEEN where q1 < q2)(
    r[q1] != r[q2] 
    /\ 
    c[q1] != c[q2]
    /\
    r[q1]+c[q1] != r[q2]+c[q2]
    /\
    r[q1]-c[q1] != r[q2]-c[q2]
  );
  
output [
  join("\n", [
    join("  ", [if exists(q in QUEEN)(fix(r[q])=i /\ fix(c[q])=j) then "*" else "-" endif | j in COL])
    | i in ROW
  ])
];