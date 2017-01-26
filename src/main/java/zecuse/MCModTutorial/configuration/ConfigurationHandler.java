package zecuse.MCModTutorial.configuration;

import java.io.File;
import net.minecraftforge.common.config.Configuration;
import net.minecraftforge.fml.client.event.ConfigChangedEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import zecuse.MCModTutorial.reference.Reference;

// This is a basic class to load/create a configuration file for the mod.
// The user can delete the file if they want to restore the defaults.
public class ConfigurationHandler
{
	public static Configuration config;
	public static boolean bool = false;
	
	public static void init(File configFile)
	{
		if(config == null)
		{
			config = new Configuration(configFile);
			loadConfiguration();
		}
	}
	
	@SubscribeEvent
	// Resync configs
	public void onConfigurationChangedEvent(ConfigChangedEvent.OnConfigChangedEvent event)
	{
		if(event.getModID().equalsIgnoreCase(Reference.MOD_ID))
			loadConfiguration();
	}
	
	// Load/create the configuration file.
	private static void loadConfiguration()
	{
		// Read in properties from the configuration file.
		bool = config.getBoolean("Boolean", Configuration.CATEGORY_GENERAL, true, "A boolean.");
		
		// Save the configuration file.
		if(config.hasChanged())
			config.save();
		
		// Output for debugging.
//		System.out.println("My config value " + bool);
	}
}
