SetFactory("Built-in");

// ---- Rectangle (Lx=12, Ly=8) ----
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 8, 0, 1.0};
Point(3) = {12, 0, 0, 1.0};
Point(4) = {12, 8, 0, 1.0};

Line(1) = {2, 1};
Line(2) = {1, 3};
Line(3) = {3, 4};
Line(4) = {4, 2};

// ---- Cylinder (center = (6,4), radius = 0.5) ----
Point(7) = {6.0, 4.0, 0, 0.5};     // center
Point(5) = {6+0.5, 4.0, 0, 0.5};
Point(6) = {6-0.5, 4.0, 0, 0.5};
Point(8) = {6.0, 4-0.5, 0, 0.5};
Point(9) = {6.0, 4+0.5, 0, 0.5};

Circle(5) = {6, 7, 8};
Circle(6) = {8, 7, 5};
Circle(7) = {5, 7, 9};
Circle(8) = {9, 7, 6};

// ---- Surface with hole ----
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {8, 5, 6, 7};
Plane Surface(1) = {1, 2};

//MeshSize {1,2,3,4,5,6,8,9} = 0.3;
MeshSize {5,6,8,9} = 0.06;  // only enforce fine size on cylinder // used 0.005 for exact mccamy solution


// ---- Extrude to H = 1 ----
ex[] = Extrude {0, 0, 1/(2*Pi)} { Surface{1}; Layers{4}; Recombine; };

Physical Volume(1) = {ex[1]};   // ensures 3D elements are written
Physical Surface(2) = {ex[0]};  // top boundary attribute
Physical Surface(3) = {ex[6], ex[7], ex[8], ex[9]};  // cylinder wall (likely)


// ---- Coarsen mesh away from cylinder in y-direction ----
lc_near = 0.01;   // size near cylinder centerline
lc_far  = 0.01;   // size near top/bottom walls
band    = 1.0;   // half-width of refined band in y
cy      = 4.0;   // cylinder center y

Field[1] = MathEval;
Field[1].F = Sprintf("%g + (%g-%g)*min(1, abs(y-%g)/%g)",
                     lc_near, lc_far, lc_near, cy, band);

// ---- Coarsen mesh away from cylinder in x-direction ----
bandx   = 1.0;   // half-width of refined band in x
cx      = 6.0;   // cylinder center x

Field[2] = MathEval;
Field[2].F = Sprintf("%g + (%g-%g)*min(1, abs(x-%g)/%g)",
                     lc_near, lc_far, lc_near, cx, bandx);

// ---- Only keep it fine near cylinder (take the larger size away from it) ----
Field[3] = Max;
Field[3].FieldsList = {1, 2};

Background Field = 3;


Mesh.MshFileVersion = 2.2;
Mesh.SaveAll = 0;
Mesh 3;
