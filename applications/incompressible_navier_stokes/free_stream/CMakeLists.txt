PROJECT(incompressible_free_stream)

EXADG_PICKUP_EXE(solver.cpp incompressible_free_stream solver)

ADD_SUBDIRECTORY(tests)