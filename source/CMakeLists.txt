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
file(GLOB_RECURSE MsFEComplex_TARGET_LIB_SRC  "*.cc") # source files
set(MsFEComplex_TARGET_LIB_SRC ${MsFEComplex_TARGET_LIB_SRC})

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