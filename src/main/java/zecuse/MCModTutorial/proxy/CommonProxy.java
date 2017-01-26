package zecuse.MCModTutorial.proxy;

import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPostInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import net.minecraftforge.fml.common.registry.GameRegistry;
import zecuse.MCModTutorial.block.GenericBlock;
import zecuse.MCModTutorial.client.handler.KeyInputHandler;
import zecuse.MCModTutorial.configuration.ConfigurationHandler;
import zecuse.MCModTutorial.init.ModBlocks;
import zecuse.MCModTutorial.init.ModItems;
import zecuse.MCModTutorial.init.ModRecipes;
import zecuse.MCModTutorial.utility.BlockUtils;
import zecuse.MCModTutorial.utility.LogHelper;

// This class defines the common methods used by both the client and server sides of the mod.
// It is abstract in the event a particular method from the interface should not be defined here,
// but instead in both the client and server proxy classes.
public abstract class CommonProxy implements IProxy
{
	// Simple log statements show the completion of each phase.
	@Override
	public void onPreInit(FMLPreInitializationEvent event)
	{
		ConfigurationHandler.init(event.getSuggestedConfigurationFile());
		ModItems.getItems().forEach(GameRegistry::register);
		for(GenericBlock block : ModBlocks.getBlocks())
		{
			GameRegistry.register(block);
			GameRegistry.register(BlockUtils.getItemBlock(block), block.getRegistryName());
		}
		
		LogHelper.info("#######################################");
		LogHelper.info("##### Pre-initialization complete #####");
		LogHelper.info("#######################################");
	}
	
	@Override
	public void onInit(FMLInitializationEvent event)
	{
		MinecraftForge.EVENT_BUS.register(new ConfigurationHandler());
		MinecraftForge.EVENT_BUS.register(new KeyInputHandler());
		ModRecipes.init();
		
		LogHelper.info("###################################");
		LogHelper.info("##### Initialization complete #####");
		LogHelper.info("###################################");
	}
	
	@Override
	public void onPostInit(FMLPostInitializationEvent event)
	{
		LogHelper.info("########################################");
		LogHelper.info("##### Post-initialization complete #####");
		LogHelper.info("########################################");
	}
}
