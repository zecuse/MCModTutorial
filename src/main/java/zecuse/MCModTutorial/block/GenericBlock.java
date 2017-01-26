package zecuse.MCModTutorial.block;

import net.minecraft.block.Block;
import net.minecraft.block.material.Material;
import net.minecraft.client.renderer.block.model.ModelResourceLocation;
import net.minecraft.item.Item;
import net.minecraftforge.client.model.ModelLoader;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;
import zecuse.MCModTutorial.creativetab.CreativeTabTutorial;
import zecuse.MCModTutorial.init.ModBlocks;
import zecuse.MCModTutorial.reference.Reference;

// This class is used as a wrapper for all other mod blocks.
// Default properties are setup here.
// Should a block require different values, they are defined in its class.
public abstract class GenericBlock extends Block
{
	private final String GENERIC_NAME;
	
	public GenericBlock(String name)
	{
		this(name, Material.ROCK);
	}
	
	public GenericBlock(String name, Material material)
	{
		super(material);
		this.setRegistryName(name);
		this.setUnlocalizedName(name);
		this.setCreativeTab(CreativeTabTutorial.TUTORIAL_TAB);
		GENERIC_NAME = name;
		ModBlocks.register(this);
	}
	
	// Retrieves "item.(MOD ID):(NAME)" replace (...) with the appropriate value.
	@Override
	public String getUnlocalizedName()
	{
		return String.format("tile.%s:%s", Reference.MOD_ID, GENERIC_NAME);
	}
	
	// This method loads the block's model into the game.
	@SideOnly(Side.CLIENT)
	public void initBlockModels()
	{
		ModelLoader.setCustomModelResourceLocation(Item.getItemFromBlock(this), 0, new ModelResourceLocation(this.getRegistryName().toString()));
	}
}
