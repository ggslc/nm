import jax
import jax.numpy as jnp
import numpy as np
import scipy
import matplotlib.pyplot as plt


#TODO: function for translating stencil into set of basis vectors


def dodgy_coo_to_csr(values, coordinates, shape, return_decomposition=False):

    a = scipy.sparse.coo_array((values, (coordinates[:,0], coordinates[:,1])), shape=shape).tocsr()

    if return_decomposition:
        return a.indptr, a.indices, a.data
    else:
        return a

def make_sparse_jacrev_fct_multiprime_no_densify(basis_vectors):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)

        #I think this is true at least...
        m_range = range(len(primals))
        rows_agg = [[] for i in m_range]
        
        #need to get rid of this loop!!
        #maybe vmap or something? Maybe it's not so bad if n(bvs)~10...
        for bv in basis_vectors:
            row_tuple = vjp_fun(bv)
            for i in m_range:
                rows_agg[i].append(row_tuple[i])
        
        rows_agg = [jnp.concatenate(rows_agg[i]) for i in m_range]

        return rows_agg

    return sparse_jacrev

def make_sparse_jacrev_fct_multiprime(n, basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)

        #I think this is true at least...
        m_range = range(len(primals))
        rows_agg = [[] for i in m_range]
        
        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row_tuple = vjp_fun(bv)
            for i in m_range:
                rows_agg[i].append(row_tuple[i])
        
        rows_agg = [jnp.concatenate(rows_agg[i]) for i in m_range]

        return rows_agg

    def densify_sparse_jac(jacrows_vec):
        m = len(basis_vectors[0]) #number of columns in matrix
        jac = jnp.zeros((n, m))

        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
            # jac = jac.at[bv_is, bv_js].set(jacrow)

        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac

def make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)
        rows = []

        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row, _ = vjp_fun(bv)
            rows.append(row)

        rows = jnp.concatenate(rows)

        # print(rows)
        return rows

    def densify_sparse_jac(jacrows_vec):
        jac = jnp.zeros((n, n))

        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
            # jac = jac.at[bv_is, bv_js].set(jacrow)

        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac



def make_sparse_jacrev_fct_sqaure_stencil_2d(basis_vectors, i_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory

    def sparse_jacrev(fun_, primals):
        
        y, vjp_fun = jax.vjp(fun_, *primals)
        rows = []

        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row, _ = vjp_fun(bv)
            rows.append(row)

        rows = jnp.concatenate(rows)

        # print(rows)
        return rows

    def densify_sparse_jac(jacrows_vec):
        jac = jnp.zeros((n, n))

        jac = jac.at[i_coord_sets, :].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac




def create_repeated_array(base_array, n):
    repetitions = int(jnp.ceil(n / len(base_array)))

    repeated_array = jnp.tile(base_array, repetitions)

    return repeated_array[:n], repetitions


def basis_vectors_etc_nonsquare(nr, nc, case_=1, fill_from_left=True, fill_from_top=True):
    """
    create basis vectors with which to carry out jacobian-vector products and
    sets of coordinates mapping the corresponding jvps to the dense jacobian.

    I can only handle filling in from top left. JC.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian
    case 4: lower tridiagonal jaconian! (not sure it's a thing but ykwim)
    case 5: pentagonal jacobian

    """

    match case_:

        case 0: ##UNTESTED
            if nc>=nr:
                basis_vectors = [jnp.ones((nr,))]
                i_coord_sets  = [jnp.concatenate(jnp.arange(nr), jnp.zeros((nc-nr,))+nr-1)] #getting those trailing zeros in there
                j_coord_sets  = [jnp.arange(nc)]
            else:
                #basis_vectors = [jnp.ones((nc,))]
                #i_coord_sets  = [jnp.arange(nc)]
                #j_coord_sets  = [jnp.arange(nc)]
                raise NotImplementedError


        case 1:
            if nc>nr:
                base_1 = np.array([1, 0, 0])
                base_2 = np.array([0, 1, 0])
                base_3 = np.array([0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    
                    is_ = np.repeat(np.arange(k, nr, 3), 3)
                    
                    if basis[0]==1:
                        is_ = is_[1:]

                    if basis[-1]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr-1)])
                    if basis[-2]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr)])
                    if basis[-3]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr+1)])

                    if basis[0]==0 and basis[1]==0:
                        is_ = np.insert(is_, 0, is_[0])

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)

            elif nc==(nr-1):
                
                base_1 = np.array([1, 0, 0])
                base_2 = np.array([0, 1, 0])
                base_3 = np.array([0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    
                    is_ = np.repeat(np.arange(k, nr, 3), 3)
                   
                    #TODO: Add functionality for things being quite different sizes.
                    #so far only works if nr-nc=1

                    if basis[0]==1:
                        is_ = is_[1:]
                    if basis[-1]==1:
                        is_ = is_[:-2]
                    if basis[-2]==1:
                        is_ = is_[:-1]

                    #remember, you need those zeros to make the things the same length!
                    if basis[0]==0 and basis[1]==0:
                        is_ = np.insert(is_, 0, is_[0])
                    

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)


            else:
                raise NotImplementedError

        case 2: ##UNTESTED
            raise NotImplementedError

        case 3: ##UNTESTED
            raise NotImplementedError
        
        case 4: ##UNTESTED
            raise NotImplementedError
        
        case 5:
            if nc==(nr-1):
                
                base_1 = np.array([1, 0, 0, 0, 0])
                base_2 = np.array([0, 1, 0, 0, 0])
                base_3 = np.array([0, 0, 1, 0, 0])
                base_4 = np.array([0, 0, 0, 1, 0])
                base_5 = np.array([0, 0, 0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3, base_4, base_5]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    is_ =  np.repeat(np.arange(k, nr, 5), 5)
                    
                    if basis[0]==1:
                        is_ = is_[2:]
                    elif basis[1]==1:
                        is_ = is_[1:]

                    if basis[-1]==1:
                        is_ = is_[:-3]
                    elif basis[-2]==1:
                        is_ = is_[:-2]
                    elif basis[-3]==1:
                        is_ = is_[:-1]

                    #remember, you need those zeros to make the things the same length!
                    if basis[0]==0 and basis[1]==0 and basis[2]==0:
                        is_ = np.insert(is_, 0, is_[0])
                        if basis[3]==0:
                            is_ = np.insert(is_, 0, is_[0])
                    if basis[-1]==0 and basis[-2]==0 and basis[-3]==0 and basis[-4]==0:
                        is_ = np.append(is_, is_[-1])

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)

            else:
                raise NotImplementedError

    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets


def basis_vectors_etc(n, case_=1):
    """
    create basis vectors with which to carry out jacobian-vector products and
    sets of coordinates mapping the corresponding jvps to the dense jacobian.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian
    case 4: lower tridiagonal jaconian! (not sure it's a thing but ykwim)
    case 5: pentagonal jacobian

    """

    match case_:

        case 0: ##UNTESTED
              basis_vectors = [jnp.ones((n,))]
              i_coord_sets  = [jnp.arange(n)]
              j_coord_sets  = [jnp.arange(n)]


        case 1:
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 3), 3)
                
                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0:
                    is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 2: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[-1]==1:
                    is_ = is_[:-1]

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 3: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 4: ##UNTESTED
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ = np.repeat(np.arange(k, n, 3), 3)

                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]

                #remember, you need those zeros to make the things the same length!
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-2]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 5:
            base_1 = np.array([1, 0, 0, 0, 0])
            base_2 = np.array([0, 1, 0, 0, 0])
            base_3 = np.array([0, 0, 1, 0, 0])
            base_4 = np.array([0, 0, 0, 1, 0])
            base_5 = np.array([0, 0, 0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3, base_4, base_5]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 5), 5)
                
                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-2]
                elif basis[-2]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0 and basis[2]==0:
                    is_ = np.insert(is_, 0, is_[0])
                    if basis[3]==0:
                        is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0 and basis[-3]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-4]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets



#Going for basis vectors with which to do a vector-jacobian product!

def oo_basis_vectors_2d_square_stencil(nr, nc, stencil_radius=2):
    #stencil_radius of 1 gives a 9-point stencil,
    #...of 2 gives a 25-point stencil
    #...of 3 gives a 49-point stencil etc

    # width of the patch of the grid surrounding each colour
    # = stencil with plus half a stencil width plus one
    # = (2*stencil_radius + 1) + stencil_radius + 1
    colour_patch_width = 3*stencil_radius + 2

    #You just have to colour the domain in patches of the same size as the stencil
    stencil_width = 2*stencil_radius + 1

    colouring_template = jnp.arange(stencil_width**2).reshape(stencil_width, -1)

    #NOTE: It might be worth saving a copy of the coords in COO and CSR formats at this
    #point. It's annoying, but maybe better than switching between using scipy.

    n_stencils_wide = int(jnp.ceil(nc/stencil_width))
    n_stencils_tall = int(jnp.ceil(nr/stencil_width))


    tiled_domain = jnp.tile(colouring_template, (n_stencils_tall, n_stencils_wide))[:nr, :nc]
    td_flat = tiled_domain.flatten()

    basis_vectors = []

    #Need to know, when you do the vjp, what the coordinates are of the resulting non-zero values too...
    j_coords = jnp.arange(nr*nc, dtype=jnp.int16)
    i_coords = []

    for colour in range(stencil_width**2):
        colour_indices = jnp.where(td_flat==colour)[0].astype(jnp.int16)
        
        bv = jnp.zeros((nr*nc), dtype=jnp.int16).at[colour_indices].set(1)
        basis_vectors.append(bv)

        colour_coordinates = jnp.zeros_like(bv, dtype=jnp.int16).at[colour_indices].set(colour_indices)

        ccs_padded = jnp.pad(colour_coordinates.reshape(nr, nc),\
                     pad_width=((stencil_radius, stencil_radius), (stencil_radius, stencil_radius)),\
                     mode='constant', constant_values=0)

        i_coordinates = jnp.zeros((nr+2*stencil_radius, nc+2*stencil_radius), dtype=jnp.int16)
        # Sum over all neighbors in the square stencil
        for dx in range(-stencil_radius, stencil_radius + 1):
            for dy in range(-stencil_radius, stencil_radius + 1):
                i_coordinates = i_coordinates + jnp.roll(ccs_padded, shift=(dx, dy), axis=(0, 1))

        i_coords.append(i_coordinates[stencil_radius:-stencil_radius,stencil_radius:-stencil_radius].flatten())

    return zip(basis_vectors, i_coords)


def sparsity_pattern(nr, nc, stencil_radius):
    bv_coordinate_pairs = oo_basis_vectors_2d_square_stencil(nr, nc, stencil_radius)

    j_coords = jnp.arange(nr*nc)
    
    
    pattern = jnp.zeros((nr*nc, nr*nc))*jnp.nan
    for bv, cp in bv_coordinate_pairs:
        pattern = pattern.at[cp, j_coords].set(1)

    plt.imshow(np.array(pattern))
    plt.show()

sparsity_pattern(50,50,3)


#basis_vectors, i_coord_sets, j_coord_sets = basis_vectors_etc(20, 5)
#print(basis_vectors)
#print(i_coord_sets)
#print(j_coord_sets)


#def colour_domain_uniform_mesh(domain, stencil_width=3):
#    #stencil_width of -1 is a special case of the 2D 5 point stencil which we can
#    #cover with 2 colours.
#
#    dim_ = len(domain.shape)
#    
#    basis_vectors = []
#    i_coord_sets = []
#    j_coord_sets = []
#
#    if dim_==1:
#        for n in range(stencil_width):
#            arr_ = np.zeros_like(domain)
#            arr_[n::stencil_width]=1
#            basis_vectors.append(arr_)
#
#            js_ = np.arange(domain.shape[0])
#            is_ = np.arange(0+n, domain.shape[0], stencil_width)
##            is_ = np.repeat(np.arange(k, domain.shape[0], stencil_width), stencil_width)
##
##            if basis[0]==1:
##                is_ = is_[1:]
##            if basis[-1]==1:
##                is_ = is_[:-1]
##
##            if basis[0]==0 and basis[1]==0:
##                is_ = np.insert(is_, 0, is_[0])
##            if basis[-1]==0 and basis[-2]==0:
##                is_ = np.append(is_, is_[-1])
##
#            i_coord_sets.append(is_)
#            j_coord_sets.append(js_)
#            
#            print(arr_)
#            print(is_)
#            print(js_)
#
#
#        return
#
#    if stencil_width==-1:
#
#        return
#
#
#    return

