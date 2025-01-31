# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cho620/cuda-assignment-1-3mpty9

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cho620/cuda-assignment-1-3mpty9

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/cho620/cuda-assignment-1-3mpty9/CMakeFiles /home/cho620/cuda-assignment-1-3mpty9//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/cho620/cuda-assignment-1-3mpty9/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named lab1

# Build rule for target.
lab1: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 lab1
.PHONY : lab1

# fast build rule for target.
lab1/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/lab1.dir/build.make CMakeFiles/lab1.dir/build
.PHONY : lab1/fast

#=============================================================================
# Target rules for targets named cudaLib

# Build rule for target.
cudaLib: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cudaLib
.PHONY : cudaLib

# fast build rule for target.
cudaLib/fast:
	$(MAKE) $(MAKESILENT) -f src/CMakeFiles/cudaLib.dir/build.make src/CMakeFiles/cudaLib.dir/build
.PHONY : cudaLib/fast

#=============================================================================
# Target rules for targets named cpuLib

# Build rule for target.
cpuLib: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cpuLib
.PHONY : cpuLib

# fast build rule for target.
cpuLib/fast:
	$(MAKE) $(MAKESILENT) -f src/CMakeFiles/cpuLib.dir/build.make src/CMakeFiles/cpuLib.dir/build
.PHONY : cpuLib/fast

lab1.o: lab1.cu.o
.PHONY : lab1.o

# target to build an object file
lab1.cu.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/lab1.dir/build.make CMakeFiles/lab1.dir/lab1.cu.o
.PHONY : lab1.cu.o

lab1.i: lab1.cu.i
.PHONY : lab1.i

# target to preprocess a source file
lab1.cu.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/lab1.dir/build.make CMakeFiles/lab1.dir/lab1.cu.i
.PHONY : lab1.cu.i

lab1.s: lab1.cu.s
.PHONY : lab1.s

# target to generate assembly for a file
lab1.cu.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/lab1.dir/build.make CMakeFiles/lab1.dir/lab1.cu.s
.PHONY : lab1.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... cpuLib"
	@echo "... cudaLib"
	@echo "... lab1"
	@echo "... lab1.o"
	@echo "... lab1.i"
	@echo "... lab1.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

