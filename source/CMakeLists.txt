###############################################################################
###############################################################################
#
# Include directory for sources
#
include_directories(${MsFEComplex_INCLUDE_DIR})
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Name all sources
#
#file(GLOB_RECURSE MsFEComplex_TARGET_LIB_SRC  "*.cc") # source files
#set(MsFEComplex_TARGET_LIB_SRC ${MsFEComplex_TARGET_LIB_SRC})
set(MsFEComplex_TARGET_LIB_SRC
	eqn_boundary_vals.cc
	eqn_coeff_A.cc
	eqn_coeff_B.cc
	eqn_coeff_R.cc
	eqn_exact_solution_lin.cc
	eqn_rhs.cc
	my_vector_tools.inst.cc
	ned_rt_basis.cc
	ned_rt_global.cc
	ned_rt_post_processor.cc
	ned_rt_ref.cc
	parameters.cc
	q_ned_basis.cc
	q_ned_global.cc
	q_ned_post_processor.cc
	q_ned_ref.cc)

print_all_args (
  ${MsFEComplex_TARGET_LIB_SRC}
 )
###############################################################################
###############################################################################


###############################################################################
###############################################################################
#
# Compile and link the sources as SHARED
#
add_library (MsFEComplex_LIBRARY SHARED ${MsFEComplex_TARGET_LIB_SRC})
DEAL_II_SETUP_TARGET(MsFEComplex_LIBRARY)

#
# Install into the DESTINATION provided by CMAKE_INSTALL_PREFIX
#
#install (TARGETS ${MsFEComplex_LIBRARY} DESTINATION ${CMAKE_INSTALL_PREFIX})

add_executable(MsFEComplex_Ned_RT "main_ned_rt.cxx")
DEAL_II_SETUP_TARGET(MsFEComplex_Ned_RT)
TARGET_LINK_LIBRARIES(MsFEComplex_Ned_RT MsFEComplex_LIBRARY)

add_executable(MsFEComplex_Q_Ned "main_q_ned.cxx")
DEAL_II_SETUP_TARGET(MsFEComplex_Q_Ned)
TARGET_LINK_LIBRARIES(MsFEComplex_Q_Ned MsFEComplex_LIBRARY)
###############################################################################
###############################################################################