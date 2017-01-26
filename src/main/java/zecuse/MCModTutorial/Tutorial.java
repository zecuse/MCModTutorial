package zecuse.MCModTutorial;

import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.SidedProxy;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPostInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import zecuse.MCModTutorial.proxy.IProxy;
import zecuse.MCModTutorial.reference.Reference;

// This class is used to load the mod into Minecraft.
@Mod(modid = Reference.MOD_ID, name = Reference.MOD_NAME, version = Reference.VERSION, guiFactory = Reference.GUI_FACTORY)
public class Tutorial
{
	// This is used by other mods to interact with this mod.
	@Mod.Instance(Reference.MOD_ID)
	public static Tutorial instance;
	
	// This is used for the client and server side of things for a mod.
	@SidedProxy(clientSide = Reference.CLIENT_PROXY, serverSide = Reference.SERVER_PROXY)
	public static IProxy proxy;
	
	// The following sections are loaded one by one by each mod in the pack during each phase.
	// This section is loaded during the pre-initialization phase of Minecraft.
	// Read configs, create blocks, items, ect.
	@Mod.EventHandler
	public void preInit(FMLPreInitializationEvent event)
	{
		proxy.onPreInit(event);
	}
	
	// This section is loaded during the initialization phase of Minecraft.
	// Setup the mod, talk to any other mods.
	@Mod.EventHandler
	public void init(FMLInitializationEvent event)
	{
		proxy.onInit(event);
	}
	
	// This section is loaded during the post-initialization phase of Minecraft.
	// Respond to other mods, finish setup based on that info.
	@Mod.EventHandler
	public void postInit(FMLPostInitializationEvent event)
	{
		proxy.onPostInit(event);
	}
}
