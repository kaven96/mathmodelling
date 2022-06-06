int: N = 30;
set of int: ROW = 1..N;
set of int: COL = 1..N;

array[ROW, COL] of var bool: x;

constraint
  forall(r in ROW)(
    exists(c in COL)(
      x[r, c]
    )
    /\
    not exists(c1, c2 in COL where c1 < c2)(
      x[r, c1] /\ x[r, c2]
    )
  );

constraint
  forall(c in COL)(
    exists(r in ROW)(
      x[r, c]
    )
    /\
    not exists(r1, r2 in ROW where r1 < r2)(
      x[r1,c] /\ x[r2,c]
    )
  );

constraint
  not exists(r1 in ROW, c1 in COL, r2 in ROW, c2 in COL where (r1+c1 = r2+c2 \/ r1-c1=r2-c2) /\ (r1 < r2 \/ c1 < c2))(
    x[r1, c1] /\ x[r2, c2]
  );


output [
  join("\n", [
    join("  ", [if fix(x[r, c]) then "*" else "-" endif | c in COL])
    | r in ROW
  ])
];