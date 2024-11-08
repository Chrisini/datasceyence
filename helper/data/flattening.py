from scipy.optimize import curve_fit

from abc import ABC, abstractmethod
import numpy as np



class Operation(ABC):
    # https://github.com/forihuelaespina/OCTant/blob/master/src/octant/op/Operation.py
    
    def __init__(self,**kwargs):
        """The class constructor.

        The class constructor. Creates an empty operation

        """
        super().__init__()

        #Initialize attributes (without decorator @property)

        #Initialize properties (with decorator @property)
        self.name = 'Operation' #The operation name
        self.operands = list() #Operands
        self.parameters = list() #Parameters
        self.result = None #Operation outputs (a list in case it is multivalued).
                           #None until executed or cleared.

        if kwargs is not None:
            for key, value in kwargs.items():
                if (key=='name'):
                    self.name = value

        return

    #Properties getters/setters
    #
    # Remember: Sphinx ignores docstrings on property setters so all
    #documentation for a property must be on the @property method
    @property
    def operands(self): #operands getter
        """
        The list of operands.

        :getter: Gets the list of operands
        :setter: Sets the list of operands.
        :type: list
        """
        return self.__operands


    @operands.setter
    def operands(self,opList): #operands setter
        #if (not isinstance(opList,(list,))):
        if type(opList) is not list:
            warnMsg = self.getClassName() + ':operands: Unexpected type. ' \
                            'Please provide operands as a list.'
            warnings.warn(warnMsg,SyntaxWarning)
        else:
            self.__operands = opList;
        return None

    @property
    def name(self): #name getter
        """
        The operation name

        :getter: Gets the operation name
        :setter: Sets the operation name.
        :type: string
        """
        return self.__name

    @name.setter
    def name(self,opName): #name setter
        #if (not isinstance(opName,(str,))):
        if type(opName) is not str:
            warnMsg = self.getClassName() + ':name: Unexpected type. ' \
                            'Operations name must be a string.'
            warnings.warn(warnMsg,SyntaxWarning)
        else:
            self.__name = opName;
        return None


    @property
    def parameters(self): #operands getter
        """
        The list of parameters.

        :getter: Gets the list of parameters
        :setter: Sets the list of parameters.
        :type: list
        """
        return self.__parameters


    @parameters.setter
    def parameters(self,opList): #operands setter
        #if (not isinstance(opList,(list,))):
        if type(opList) is not list:
            warnMsg = self.getClassName() + ':parameters: Unexpected type. ' \
                            'Please provide operands as a list.'
            warnings.warn(warnMsg,SyntaxWarning)
        else:
            self.__parameters = opList;
        return None


    @property
    def result(self): #result getter
        """
        The list of results.

        This is a read only property. There is no setter method.

        :getter: Gets the list of results
        :setter: Sets the list of results
        :type: list
        """
        return self.__result


    @result.setter
    def result(self,rList): #result setter
        self.__result = rList;
        return None


    #Private methods
    def __str__(self):
        tmp = '['
        for x in self.operands:
            tmp += format(x) + ','
        tmp+=']'
        s = '<' + self.getClassName() + '([' \
            + 'name: ' + self.name + ';' \
            + ' operands: ' + tmp + '])>'
        return s

    #Public methods
    def getClassName(self):
        """Get the class name as a string.

        Get the class name as a string.

        :returns: The class name.
        :rtype: string
        """
        return type(self).__name__

    def addOperand(self,op,i=None):
        """
        Add a new operand.

        :param op: The operand.
        :type op: object
        :param i: (optional) The operand order. If given it may shift the
            order of other operands already set. If not given, the operand
            is appended at the end of the list of operands.
        :type op: int
        :return: None
        """
        if (i is None):
            self.__operands.append(op)
        else:
            self.__operands.insert(i,op)
        return None

    def setOperand(self,op,i):
        """
        Set an operand; substitutes an existing operand with a new one.

        Calling setOperand when the :py:attr:`i`-th operand has not been
        previously set will result in an out-of-range error.

        :param op: The new operand.
        :type op: object
        :param i: The operand order. Operand index is zero-base i.e. the
            first operand occupies i=0
        :type op: int
        :return: None
        """
        self.__operands[i] = op
        return None

    def addParameter(self,param,i=None):
        """
        Add a new parameter.

        :param op: The parameter.
        :type op: object
        :param i: (optional) The paremeter order. If given it may shift the
            order of other parameters already set. If not given, the parameter
            is appended at the end of the list of parameters.
        :type op: int
        :return: None
        """
        if (i is None):
            self.__parameters.append(op)
        else:
            self.__parameters.insert(i,op)
        return None

    def setParameter(self,op,i):
        """
        Set a parameter; substitutes an existing parameter with a new one.

        Calling setParameter when the :py:attr:`i`-th parameter has not been
        previously set will result in an out-of-range error.

        :param op: The new operand.
        :type op: object
        :param i: The operand order. Operand index is zero-base i.e. the
            first operand occupies i=0
        :type op: int
        :return: None
        """
        self.__operands[i] = op
        return None

    def arity(self):
        """Gets the operation arity (number of operands).

        :return: The operation arity
        :rtype: int
        """
        return len(self.__operands)

    def clear(self):
        """
        Clears the operands; Removes all operands.

        :return: None
        """
        self.__operands = list()
        return None

    #@abstractmethod
    def execute(self,*args,**kwargs):
        """Executes the operation on the operands.

        This is an abstract method. Executes the operation on the .operands
        and stores the outcome in .result

        Operation meta-parameters may be also passed.

        :returns: Result of executing the operation.
        :rtype: Type of result -depends on subclass implementation-.
        """
        pass
    
    
class OpScanFlatten(Operation):
    """A flattening operation for :class:`data.OCTscan`.
    
    A flattening operation for :class:`data.OCTscan`.

    The operation represented by this class rectifies an OCT scan.
    
    .. seealso:: None
    .. note:: None
    .. todo:: None
        
    """
 
    #Private class attributes shared by all instances
    
    #Class constructor
    def __init__(self, image, mask=None):
        #Call superclass constructor
        super().__init__()
        
        #Set the operation name
        self.name = "Flattening"
        
        self.__deformationMap = None
        
        self.image = image
        self.mask = mask
        
        return
    
    
    @property
    def deformationMap(self): #name getter
        """
        A logical name for the study.
        
        This is a read only property.
        
        :getter: Gets the deformationMap associated to the last flattening.
        :type: str
        """
        return self.__deformationMap
    


    @staticmethod
    def fittingQuadraticModel(x, a, b, c):
        #quadratic model for curve optimization
        return a * x*x + b*x + c

    
    #Public methods
    def execute(self,*args,**kwargs):
        """Executes the operation on the :py:attr:`operands`.
        
        Executes the operation on the :py:attr:`operands` and stores the outcome
        in :py:attr:`result`. Preload operands using
        :func:`Operation.addOperand()`.
        
        :returns: Result of executing the operation.
        :rtype: :class:`data.OCTscan`
        """
        #print(self._getClasName(),": flattening: Starting flattening")
        
        #Ensure the operand has been set.
        
        
        
        #Check whether the image is in RGB (ndim=3) or in grayscale (ndim=2)
        #and convert to grayscale if necessary
        if self.image.ndim == 2:
            #Dimensions are only width and height. The image is already in grayscale.
            I2=self.image
        elif self.image.ndim == 3:
            #Image is in RGB. Convert.
            I2=color.rgb2gray(self.image);
        else: #Unexpected case. Return warning
            print(self._getClasName(),": Unexpected image shape.")
            self.result = self.image
            return self.result
        
        aux = np.argmax(I2, axis=0)
        mg = np.mean(aux)
        sdg = np.std(aux)
        markers = []
        remover =[]
        x0 = np.arange(len(aux))
        
        for i in range(0,len(aux)):
            if mg - 3*sdg <= aux[i] <= mg +3*sdg: 
                markers+= [aux[i]]
            else:
                remover+= [i]
                
        x=np.delete(x0,remover)
        
        
        
        modelCoeffs, pcov = curve_fit(self.fittingQuadraticModel, x, markers, \
                                    method = 'dogbox', loss = 'soft_l1')
        
        #print(pcov)
        #print(markers)
        
        a = self.fittingQuadraticModel(x0, *modelCoeffs)
        
        #print(a)
        
        shift = np.max(a)
        flat  = shift-a
        flat  = np.round(flat)
        flat  = np.ravel(flat).astype(int)
        self.__deformationMap = flat
        
        newgray = I2
        for i in range(0,len(a)):
            newgray[:,i] = np.roll(I2[:,i], flat[i], axis=0)

        self.result = newgray
        
        # mask part
        try:
            #print(self.__deformationMap)

            self.new_mask = []
            if self.mask is not None:
                for layer in self.mask:
                    self.new_mask.append(layer + self.__deformationMap)
        except Exception as e:
            print("no mask possible")
            return self.result, []
        
        
        
            
        return self.result, self.new_mask


    def applyOperation(self, scanA):
        """Apply the current flattening to the given scan.
        
        Instead of calculating the fitting again needed for the
        flattening, this method applies a known fitted quadratic model to
        the given parameters.
        
        The result is NOT stored in :py:attr:`result`.
        
        :param scanA: Image to flatten.
        :type scanA: :class:`data.OCTscan`
        :returns: Result of repeating the last flattening operation onto
             parameter scanA.
        :rtype: :class:`data.OCTscan`
        """
            #scanA=scanA.data
        newgray = scanA
        for i in range(0,len(self.deformationMap)):
            newgray[:,i] = np.roll(scanA[:,i], self.deformationMap[i], axis=0)
        return newgray