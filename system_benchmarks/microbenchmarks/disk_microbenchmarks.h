# pragma once
#include <string>

/**************************************************** 
 * disk_write_speed -- read and write to disk 
 *		speed
 *
 * Params: filename, name of the file to be made,
 *			use same for disk_read_speed
 *
 * Return: GB/s
 ****************************************************/ 
double disk_write_speed(std::string filename);

/**************************************************** 
 * disk_read_speed -- read speed from disk
 *
 * Params: filename, name of the file to be read 
 *			from, must be same as disk_write_speed
 * 
 * Returns: GB/s
 ****************************************************/ 
double disk_read_speed(std::string filename);
