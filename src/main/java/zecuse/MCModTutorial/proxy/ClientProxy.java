package zecuse.MCModTutorial.proxy;

import net.minecraftforge.fml.client.registry.ClientRegistry;
import net.minecraftforge.fml.common.event.FMLPreInitializationEvent;
import zecuse.MCModTutorial.block.GenericBlock;
import zecuse.MCModTutorial.client.settings.KeyBinds;
import zecuse.MCModTutorial.init.ModBlocks;
import zecuse.MCModTutorial.init.ModItems;
import zecuse.MCModTutorial.item.GenericItem;

// This class defines the methods to be used by the client side of the mod.
public class ClientProxy extends CommonProxy
{
	@Override
	public void onPreInit(FMLPreInitializationEvent event)
	{
		super.onPreInit(event);
		this.registerKeyBindings();
		ModItems.getItems().forEach(GenericItem::initItemModels);
		ModBlocks.getBlocks().forEach(GenericBlock::initBlockModels);
	}

	@Override
	public void registerKeyBindings()
	{
		ClientRegistry.registerKeyBinding(KeyBinds.charge);
		ClientRegistry.registerKeyBinding(KeyBinds.release);
	}
}
