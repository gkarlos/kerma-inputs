message(STATUS "Configuring Polybench")

add_custom_target( PolybenchGenIR ALL)
add_custom_command( TARGET PolybenchGenIR
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMAND ./kerma-make.sh -ir -silent)

message(STATUS "Configuring Polybench -- done")