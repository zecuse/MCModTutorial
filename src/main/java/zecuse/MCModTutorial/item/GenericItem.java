package zecuse.MCModTutorial.item;

import net.minecraft.client.renderer.block.model.ModelResourceLocation;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraftforge.client.model.ModelLoader;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;
import zecuse.MCModTutorial.creativetab.CreativeTabTutorial;
import zecuse.MCModTutorial.init.ModItems;
import zecuse.MCModTutorial.reference.Reference;

// This class is used as a wrapper for all other mod items.
// Default properties are setup here.
// Should an item require different values, they are defined in its class.
public abstract class GenericItem extends Item
{
	private final String GENERIC_NAME;
	
	public GenericItem(String name)
	{
		super();
		this.setRegistryName(name);
		this.setUnlocalizedName(name);
		this.setMaxStackSize(32);
		this.setCreativeTab(CreativeTabTutorial.TUTORIAL_TAB);
		GENERIC_NAME = name;
		ModItems.register(this);
	}
	
	// Retrieves "item.(MOD ID):(NAME)" replace (...) with the appropriate value.
	@Override
	public String getUnlocalizedName()
	{
		return String.format("item.%s:%s", Reference.MOD_ID, GENERIC_NAME);
	}
	
	@Override
	public String getUnlocalizedName(ItemStack itemStack)
	{
		return String.format("item.%s:%s", Reference.MOD_ID, GENERIC_NAME);
	}
	
	// This method loads the item's model into the game.
	@SideOnly(Side.CLIENT)
	public void initItemModels()
	{
		ModelLoader.setCustomModelResourceLocation(this, 0, new ModelResourceLocation(this.getRegistryName().toString()));
	}
}
