package zecuse.MCModTutorial.proxy;

import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPostInitializationEvent;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;

// This interface declares the methods to be implemented on the client and server sides of the mod.
public interface IProxy
{
	void onPreInit(FMLPreInitializationEvent event);

   void onInit(FMLInitializationEvent event);

   void onPostInit(FMLPostInitializationEvent event);
   
   void registerKeyBindings();
}
