package zecuse.MCModTutorial.utility;

import net.minecraft.block.Block;
import net.minecraft.item.ItemBlock;

public class BlockUtils
{
	public static ItemBlock getItemBlock(Block block)
	{
		return new ItemBlock(block);
	}
}
