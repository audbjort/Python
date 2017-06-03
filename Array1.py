# Given a sorted array of positive integers with an empty spot (zero) at the end insert an element in sorted order
    

from array import *
import string

print """Given a sorted array of positive integers with an empty spot (zero) at the end insert an element in sorted order """


def add_element_to_array(my_array, new_element):
  
  my_array.append(new_element)
  sorted_array = sorted(my_array)
     
  return sorted_array
  
  

def main():
   array_ex = array('i',[1,2,5,6,7,99])
   print 'Original array'
   for x in array_ex:
     print x
  
   print 'New array'
   sorted_array = add_element_to_array(array_ex, 4)
   for x in sorted_array:
     print x 
 
 
if __name__ == '__main__':
  main()    
