package zecuse.MCModTutorial.utility;

import org.apache.logging.log4j.Level;
import net.minecraftforge.fml.common.FMLLog;
import zecuse.MCModTutorial.reference.Reference;

// This class is used to log all relevant events in the log file.
// It's methods are listed in order of log4j's level precedence.
// If the logging level is set to 4, then only info to off will be logged. 
public class LogHelper
{
	public static void log(Level logLevel, Object obj)
	{
		FMLLog.log(Reference.MOD_ID, logLevel, obj.toString());
	}
	
	public static void off(Object obj)
	{
		log(Level.OFF, obj);
	}
	
	public static void fatal(Object obj)
	{
		log(Level.FATAL, obj);
	}
	
	public static void error(Object obj)
	{
		log(Level.ERROR, obj);
	}
	
	public static void warn(Object obj)
	{
		log(Level.WARN, obj);
	}
	
	public static void info(Object obj)
	{
		log(Level.INFO, obj);
	}
	
	public static void debug(Object obj)
	{
		log(Level.DEBUG, obj);
	}
	
	public static void trace(Object obj)
	{
		log(Level.TRACE, obj);
	}
	
	public static void all(Object obj)
	{
		log(Level.ALL, obj);
	}
}
